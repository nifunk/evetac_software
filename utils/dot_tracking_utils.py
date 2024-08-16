import torch
import numpy as np
import onnxruntime as rt
import onnx
from onnxconverter_common import float16
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
import pickle
import scipy
import copy
import os
from datetime import datetime

class DotTrackingOnnxModel(torch.nn.Module):
    def __init__(self, precompute_grid, pairwise_dists_mask, pairwise_dists, correction,
                                     dot_regularization_factor):
        super().__init__()
        # self.calib_centers = calib_center.cpu()
        self.precompute_grid = precompute_grid.cpu()
        self.pairwise_dists_mask = pairwise_dists_mask.cpu()
        self.pairwise_dists = pairwise_dists.cpu()
        self.correction = correction.cpu()
        self.dot_regularization_factor = dot_regularization_factor

    def forward(self, events_x, events_y, calib_center):
        dx = torch.sub(events_x.long(), calib_center[:, 1].view(-1, 1)).long()
        dy = torch.sub(events_y.long(), calib_center[:, 0].view(-1, 1)).long()

        combined = (torch.clip(torch.stack((dx, dy), dim=1), -50, 50) + 50)
        test = self.precompute_grid[combined[:, 0], combined[:, 1], :]
        update_dx_fast = torch.sum(test[:, :, 0], dim=1)
        update_dy_fast = torch.sum(test[:, :, 1], dim=1)

        update_or_not = update_dx_fast != 0
        # REGULARIZER:
        dx_centers = torch.sub(calib_center[:, 1], calib_center[:, 1].view(-1, 1))
        dy_centers = torch.sub(calib_center[:, 0], calib_center[:, 0].view(-1, 1))

        selected_dxs = torch.mul(dx_centers, self.pairwise_dists_mask)
        selected_dys = torch.mul(dy_centers, self.pairwise_dists_mask)
        radi = selected_dxs ** 2 + selected_dys ** 2 - self.pairwise_dists ** 2
        dtx_centers = 4 * torch.mul(dx_centers, radi)

        dty_centers = 4 * torch.mul(dy_centers,
                                    radi)
        center_dx_update = torch.mul(self.correction, torch.mul(update_or_not,
                                                           torch.sum(dtx_centers,
                                                                     axis=1)))  # only move if there are events
        center_dy_update = torch.mul(self.correction, torch.mul(update_or_not, torch.sum(dty_centers, axis=1)))

        calib_center[:, 1] = calib_center[:, 1] - 200 * 0.000015 * torch.clamp(update_dx_fast, -400,
                                                                                 400) + self.dot_regularization_factor * 0.00000025 * center_dx_update
        calib_center[:, 0] = calib_center[:, 0] - 200 * 0.000015 * torch.clamp(update_dy_fast, -400,
                                                                                 400) + self.dot_regularization_factor * 0.00000025 * center_dy_update

        return calib_center

class DotTrackingOnnxModelFilter(torch.nn.Module):
    def __init__(self, precompute_grid, pairwise_dists_mask, pairwise_dists, correction,
                                     dot_regularization_factor, regularize_tracking, downsample_factor_internally):
        super().__init__()
        # self.calib_centers = calib_center.cpu()
        self.precompute_grid = precompute_grid.cpu()
        self.pairwise_dists_mask = pairwise_dists_mask.cpu()
        self.pairwise_dists = pairwise_dists.cpu()
        self.correction = correction.cpu()
        self.dot_regularization_factor = dot_regularization_factor
        self.regularize_tracking = regularize_tracking
        self.threshold_update = 10 + 3*(downsample_factor_internally-1)

    def forward(self, events_x, events_y, calib_center):
        dx = torch.sub(events_x.long(), calib_center[:, 1].view(-1, 1)).long()
        dy = torch.sub(events_y.long(), calib_center[:, 0].view(-1, 1)).long()

        test = self.precompute_grid[torch.clip(dx, -50, 50) + 50, torch.clip(dy, -50, 50) + 50, :]

        decider = torch.sum(torch.sum(test!=0, dim=2), dim=1) >= self.threshold_update
        update_dx_fast = torch.sum(test[:, :, 0], dim=1)
        update_dy_fast = torch.sum(test[:, :, 1], dim=1)

        if (self.regularize_tracking):
            # update_or_not = update_dx_fast != 0
            # REGULARIZER:
            dx_centers = torch.sub(calib_center[:, 1], calib_center[:, 1].view(-1, 1))
            dy_centers = torch.sub(calib_center[:, 0], calib_center[:, 0].view(-1, 1))

            selected_dxs = torch.mul(dx_centers, self.pairwise_dists_mask)
            selected_dys = torch.mul(dy_centers, self.pairwise_dists_mask)
            radi = selected_dxs ** 2 + selected_dys ** 2 - self.pairwise_dists ** 2
            dtx_centers = 4 * torch.mul(dx_centers, radi)

            dty_centers = 4 * torch.mul(dy_centers,
                                        radi)
            center_dx_update = torch.mul(self.correction, torch.sum(dtx_centers,
                                                                         axis=1))  # only move if there are events
            center_dy_update = torch.mul(self.correction, torch.sum(dty_centers, axis=1))

            calib_center[:, 1] = calib_center[:, 1] - 200 * 0.000015 * torch.mul(decider , torch.clamp(update_dx_fast, -400,
                                                                                     400) - self.dot_regularization_factor * 0.00000025 * center_dx_update)
            calib_center[:, 0] = calib_center[:, 0] - 200 * 0.000015 * torch.mul(decider, torch.clamp(update_dy_fast, -400,
                                                                                     400) - self.dot_regularization_factor * 0.00000025 * center_dy_update)
        else:
            calib_center[:, 1] = calib_center[:, 1] - 200 * 0.000015 * torch.mul(decider , torch.clamp(update_dx_fast, -400,
                                                                                     400))
            calib_center[:, 0] = calib_center[:, 0] - 200 * 0.000015 * torch.mul(decider, torch.clamp(update_dy_fast, -400,
                                                                                     400))

        return calib_center


class DotTrackingOnnxModelFilterwNumEvents(torch.nn.Module):
    def __init__(self, precompute_grid, pairwise_dists_mask, pairwise_dists, correction,
                                     dot_regularization_factor, regularize_tracking, radius):
        super().__init__()
        # self.calib_centers = calib_center.cpu()
        self.precompute_grid = precompute_grid.cpu()
        self.pairwise_dists_mask = pairwise_dists_mask.cpu()
        self.pairwise_dists = pairwise_dists.cpu()
        self.correction = correction.cpu()
        self.dot_regularization_factor = dot_regularization_factor
        self.regularize_tracking = regularize_tracking
        self.radius = radius

    def forward(self, events_x, events_y, calib_center):
        dx = torch.sub(events_x.long(), calib_center[:, 1].view(-1, 1)).long()
        dy = torch.sub(events_y.long(), calib_center[:, 0].view(-1, 1)).long()

        vicinity = torch.logical_and(torch.abs(dx) < self.radius, torch.abs(dy) < self.radius)
        num_events_per_dot = torch.sum(vicinity, dim=1)

        test = self.precompute_grid[torch.clip(dx, -50, 50) + 50, torch.clip(dy, -50, 50) + 50, :]

        decider = torch.sum(torch.sum(test!=0, dim=2), dim=1) >= 10
        update_dx_fast = torch.sum(test[:, :, 0], dim=1)
        update_dy_fast = torch.sum(test[:, :, 1], dim=1)

        if (self.regularize_tracking):
            # update_or_not = update_dx_fast != 0
            # REGULARIZER:
            dx_centers = torch.sub(calib_center[:, 1], calib_center[:, 1].view(-1, 1))
            dy_centers = torch.sub(calib_center[:, 0], calib_center[:, 0].view(-1, 1))

            selected_dxs = torch.mul(dx_centers, self.pairwise_dists_mask)
            selected_dys = torch.mul(dy_centers, self.pairwise_dists_mask)
            radi = selected_dxs ** 2 + selected_dys ** 2 - self.pairwise_dists ** 2
            dtx_centers = 4 * torch.mul(dx_centers, radi)

            dty_centers = 4 * torch.mul(dy_centers,
                                        radi)
            center_dx_update = torch.mul(self.correction, torch.sum(dtx_centers,
                                                                         axis=1))  # only move if there are events
            center_dy_update = torch.mul(self.correction, torch.sum(dty_centers, axis=1))

            calib_center[:, 1] = calib_center[:, 1] - 200 * 0.000015 * torch.mul(decider , torch.clamp(update_dx_fast, -400,
                                                                                     400) - self.dot_regularization_factor * 0.00000025 * center_dx_update)
            calib_center[:, 0] = calib_center[:, 0] - 200 * 0.000015 * torch.mul(decider, torch.clamp(update_dy_fast, -400,
                                                                                     400) - self.dot_regularization_factor * 0.00000025 * center_dy_update)
        else:
            calib_center[:, 1] = calib_center[:, 1] - 200 * 0.000015 * torch.mul(decider , torch.clamp(update_dx_fast, -400,
                                                                                     400))
            calib_center[:, 0] = calib_center[:, 0] - 200 * 0.000015 * torch.mul(decider, torch.clamp(update_dy_fast, -400,
                                                                                     400))

        return calib_center, num_events_per_dot




class DotTrackingOnnxModelUnregularized(torch.nn.Module):
    def __init__(self, precompute_grid, pairwise_dists_mask, pairwise_dists, correction,
                                     dot_regularization_factor):
        super().__init__()
        # self.calib_centers = calib_center.cpu()
        self.precompute_grid = precompute_grid.cpu()
        self.pairwise_dists_mask = pairwise_dists_mask.cpu()
        self.pairwise_dists = pairwise_dists.cpu()
        self.correction = correction.cpu()
        self.dot_regularization_factor = dot_regularization_factor

    def forward(self, events_x, events_y, calib_center):
        dx = torch.sub(events_x.long(), calib_center[:, 1].view(-1, 1)).long()
        dy = torch.sub(events_y.long(), calib_center[:, 0].view(-1, 1)).long()

        combined = (torch.clip(torch.stack((dx, dy), dim=1), -50, 50) + 50)
        test = self.precompute_grid[combined[:, 0], combined[:, 1], :]
        update_dx_fast = torch.sum(test[:, :, 0], dim=1)
        update_dy_fast = torch.sum(test[:, :, 1], dim=1)

        calib_center[:, 1] = calib_center[:, 1] - 200 * 0.000015 * torch.clamp(update_dx_fast, -400,
                                                                                 400)
        calib_center[:, 0] = calib_center[:, 0] - 200 * 0.000015 * torch.clamp(update_dy_fast, -400,
                                                                                 400)

        return calib_center

class DotTrackOnnx:
    def __init__(self, calibration, gpu=False, regularize_tracking=True, return_num_events=False, downsample_factor_internally=1):
        self.gpu = gpu
        self.downsample_factor_internally = downsample_factor_internally
        self.return_num_events = return_num_events
        # this controls whether the tracker is regularized or not (i.e. whether we put a constraint on the dots movement or not)
        self.regularize_tracking = regularize_tracking
        self.dot_scale_factor = 1.1 # additional scaling needed when dealing with the smaller dots
        self.first_reinit = False
        self.list_of_calibrations = []
        self.calibration = calibration

        self.list_of_calibrations.append(self.calibration)
        self.calib_params = pickle.load(open(self.calibration, "rb"))
        self.initialize_tracking()

        self.normal_dx_update_default = np.zeros_like(self.calib_params[1][:].reshape(-1))
        self.normal_dy_update_default = np.zeros_like(self.calib_params[1][:].reshape(-1))
        if (torch.max(torch.sum(self.pairwise_dists_mask, axis=1))!=8):
            print ("The maximum number of neighbors is not 8 - likely there is something wrong with the regularization in the tracker!")
        self.correction = 8.0 / torch.sum(self.pairwise_dists_mask, axis=1)
        self.precompute_update()

        self.export_dot_track_onnx()

        if (self.gpu):
            model = onnx.load(os.path.dirname(os.path.realpath(__file__)) + "/onnx_models/" + "dot_track.onnx")
            model_fp16 = float16.convert_float_to_float16(model)
            onnx.save(model_fp16, os.path.dirname(os.path.realpath(__file__)) + "/onnx_models/" + "dot_track_fp16.onnx")
            self.sess = rt.InferenceSession(
                os.path.dirname(os.path.realpath(__file__)) + "/onnx_models/" + "dot_track_fp16.onnx",
                providers=['CUDAExecutionProvider'])

        else:
            sess_options = rt.SessionOptions()
            sess_options.intra_op_num_threads = 3
            sess_options.inter_op_num_threads = 4
            sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL
            self.sess = rt.InferenceSession(os.path.dirname(os.path.realpath(__file__)) + "/onnx_models/" + "dot_track.onnx", sess_options, providers=['CPUExecutionProvider'])





    def store_current_calibration(self):
        now = datetime.now()
        date_now = now.strftime("%d_%m_%Y")
        time_now = now.strftime("%H_%M_%S")

        if not(self.first_reinit):
            self.first_reinit = True
            self.filename_static = self.calibration.rsplit(".")

        curr_calibration = self.filename_static[0] + "_" + date_now + "_" + time_now + "." + self.filename_static[1]
        self.list_of_calibrations.append(curr_calibration)

        file = open(curr_calibration, 'wb')

        pickle.dump((np.fliplr(self.calib_center), self.calib_radius), file)
        file.close()
        print("Storing current calibration at: ", curr_calibration)

    def reinitialize(self, calib_center=None):
        # reinitializes to the last stored calibration
        self.calibration = self.list_of_calibrations[-1]

        self.calib_params = pickle.load(open(self.calibration, "rb"))
        self.initialize_tracking(calib_center=calib_center)

        self.correction = 8.0 / torch.sum(self.pairwise_dists_mask, axis=1)
        self.precompute_update()

        self.export_dot_track_onnx()

        if (self.gpu):
            model = onnx.load(os.path.dirname(os.path.realpath(__file__)) + "/onnx_models/" + "dot_track.onnx")
            model_fp16 = float16.convert_float_to_float16(model)
            onnx.save(model_fp16, os.path.dirname(os.path.realpath(__file__)) + "/onnx_models/" + "dot_track_fp16.onnx")
            self.sess = rt.InferenceSession(
                os.path.dirname(os.path.realpath(__file__)) + "/onnx_models/" + "dot_track_fp16.onnx",
                providers=['CUDAExecutionProvider'])

        else:
            sess_options = rt.SessionOptions()
            sess_options.intra_op_num_threads = 3
            sess_options.inter_op_num_threads = 4
            sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL
            self.sess = rt.InferenceSession(os.path.dirname(os.path.realpath(__file__)) + "/onnx_models/" + "dot_track.onnx", sess_options, providers=['CPUExecutionProvider'])


    def initialize_tracking(self, calib_center=None):
        # dot regularization factor:
        self.dot_regularization_factor = 1.5*200
        # perform a couple of necessary computations, give the calibration was loaded
        if (calib_center is None):
            self.calib_center = np.fliplr((self.calib_params[0][:, :].reshape(-1, 2)))
            self.calib_center_init = torch.fliplr(torch.from_numpy((self.calib_params[0][:, :].reshape(-1, 2)))).cpu()
            self.pairwise_dists = scipy.spatial.distance.cdist(self.calib_params[0][:, :], self.calib_params[0][:, :])
        else:
            self.calib_center = np.fliplr((calib_center.reshape(-1, 2)))
            self.calib_center_init = torch.fliplr(torch.from_numpy(calib_center.reshape(-1, 2))).cpu()
            self.pairwise_dists = scipy.spatial.distance.cdist(calib_center, calib_center)

        self.pairwise_dists_mask = self.pairwise_dists[:,:] < 105
        np.fill_diagonal(self.pairwise_dists_mask, 0)
        self.pairwise_dists = np.multiply(self.pairwise_dists_mask, self.pairwise_dists)
        self.pairwise_dists_mask = torch.from_numpy(self.pairwise_dists_mask)
        self.pairwise_dists = torch.from_numpy(self.pairwise_dists)

        self.calib_center_original = (copy.deepcopy(self.calib_center))
        self.calib_radius = self.calib_params[1][:].reshape(-1, 1)
        self.mean_radius = np.mean(self.calib_radius) #* 0.75 this factor was needed when transitioning from big to small gels
        self.calib_radius[:] = self.mean_radius
        self.num_events_per_dot = np.zeros_like(self.calib_radius)

    def precompute_update(self):
        num_points = 50
        self.precompute_grid = torch.zeros((2 * num_points + 1, 2 * num_points + 1, 2)).cuda()
        # create grid indices
        x = torch.linspace(-num_points, num_points, 2 * num_points + 1).cuda()
        y = torch.linspace(-num_points, num_points, 2 * num_points + 1).cuda()
        x_grid, y_grid = torch.meshgrid(x, y)
        events_x = x_grid.reshape(-1)
        events_y = y_grid.reshape(-1)

        dx = events_x
        dy = events_y
        dist = torch.sqrt(dx ** 2 + dy ** 2)

        decision_filter = torch.logical_or((torch.greater(dist, self.mean_radius + 0)),
                                           (torch.le(dist, self.mean_radius - 12.5*self.dot_scale_factor)))

        dx[decision_filter] = 0
        dy[decision_filter] = 0
        # clip the distance as we divide by it to avoid too large gradients
        dist[dist < 1] = 1.0
        dist[decision_filter] = 10.0

        dtx = 2 * torch.sub(dx, torch.mul(self.mean_radius * 1.25, torch.divide(dx, dist)))
        dty = 2 * torch.sub(dy, torch.mul(self.mean_radius * 1.25, torch.divide(dy, dist)))

        self.precompute_grid[:, :, 0] = dtx.reshape(2 * num_points + 1, 2 * num_points + 1)
        self.precompute_grid[:, :, 1] = dty.reshape(2 * num_points + 1, 2 * num_points + 1)

    def export_dot_track_onnx(self):
        # export the network to onnx
        # if standard - i.e., only return dot locations:
        if not(self.return_num_events):
            net = DotTrackingOnnxModelFilter(self.precompute_grid, self.pairwise_dists_mask, self.pairwise_dists, self.correction,
                                             self.dot_regularization_factor, self.regularize_tracking, self.downsample_factor_internally)

            events_x = torch.from_numpy(np.zeros((2), dtype=np.int16))
            events_y = torch.from_numpy(np.zeros((2), dtype=np.int16))

            torch.onnx.export(net,  # model being run
                              (events_x, events_y, self.calib_center_init),  # model input (or a tuple for multiple inputs)
                              os.path.dirname(os.path.realpath(__file__)) + "/onnx_models/" + "dot_track.onnx",  # where to save the model (can be a file or file-like object)
                              export_params=True,  # store the trained parameter weights inside the model file
                              opset_version=12,  # the ONNX version to export the model to
                              do_constant_folding=True,  # whether to execute constant folding for optimization
                              input_names=['input1', 'input2', 'input3'],  # the model's input names
                              output_names=['output1'],
                              dynamic_axes={'input1': {0: 'batch_size'},  # variable length axes
                                            'input2': {0: 'batch_size'}})
        else:
            net = DotTrackingOnnxModelFilterwNumEvents(self.precompute_grid, self.pairwise_dists_mask, self.pairwise_dists, self.correction,
                                             self.dot_regularization_factor, self.regularize_tracking, 20)
            events_x = torch.from_numpy(np.zeros((2), dtype=np.int16))
            events_y = torch.from_numpy(np.zeros((2), dtype=np.int16))

            torch.onnx.export(net,  # model being run
                              (events_x, events_y, self.calib_center_init),
                              # model input (or a tuple for multiple inputs)
                              os.path.dirname(os.path.realpath(__file__)) + "/onnx_models/" + "dot_track.onnx",
                              # where to save the model (can be a file or file-like object)
                              export_params=True,  # store the trained parameter weights inside the model file
                              opset_version=12,  # the ONNX version to export the model to
                              do_constant_folding=True,  # whether to execute constant folding for optimization
                              input_names=['input1', 'input2', 'input3'],  # the model's input names
                              output_names=['output1', 'output2'],
                              dynamic_axes={'input1': {0: 'batch_size'},  # variable length axes
                                            'input2': {0: 'batch_size'}})

    def track(self, events_x, events_y):
        ort_inputs = {'input1': events_x, 'input2': events_y, 'input3': self.calib_center}
        ort_outs = self.sess.run(None, ort_inputs)
        self.calib_center = ort_outs[0]
        if self.return_num_events:
            return self.calib_center, ort_outs[1]
        else:
            return self.calib_center

    def get_dot_locations(self):
        return self.calib_center

    def get_num_events_per_dot(self):
        return self.num_events_per_dot



class DotTrackingModelWoPrecompute(torch.nn.Module):
    def __init__(self, pairwise_dists_mask, pairwise_dists, correction,
                                     dot_regularization_factor, calib_radius):
        self.pairwise_dists_mask = pairwise_dists_mask.cpu()
        self.pairwise_dists = pairwise_dists.cpu()
        self.correction = correction.cpu()
        self.dot_regularization_factor = dot_regularization_factor
        self.calib_radius = calib_radius

    def forward(self, events_x, events_y, calib_center):
        dx = torch.sub(events_x, calib_center[:, 1].view(-1, 1))
        dy = torch.sub(events_y, calib_center[:, 0].view(-1, 1))
        dist = torch.sqrt(dx ** 2 + dy ** 2)

        decision_filter = torch.logical_or((torch.greater(dist, self.calib_radius + 0)),
                                           (torch.le(dist, self.calib_radius - 12.5)))

        dx[decision_filter] = 0
        dy[decision_filter] = 0
        dist[dist < 1] = 1.0

        dist[decision_filter] = 10.0  # deactivate all the distances
        dtx = 2 * torch.sub(dx, torch.mul(self.calib_radius * 1.25, torch.divide(dx, dist)))
        dty = 2 * torch.sub(dy, torch.mul(self.calib_radius * 1.25, torch.divide(dy, dist)))
        normal_dx_update = torch.sum(dtx, axis=1)
        normal_dy_update = torch.sum(dty, axis=1)
        update_or_not = normal_dx_update != 0

        # REGULARIZER:
        dx_centers = torch.sub(self.calib_center[:, 1], self.calib_center[:, 1].view(-1, 1))
        dy_centers = torch.sub(self.calib_center[:, 0], self.calib_center[:, 0].view(-1, 1))
        dist_centers = torch.sqrt(dx_centers ** 2 + dy_centers ** 2)
        selected_dxs = torch.mul(dx_centers, self.pairwise_dists_mask)
        selected_dys = torch.mul(dy_centers, self.pairwise_dists_mask)
        selected_center_dists = torch.mul(dist_centers, self.pairwise_dists_mask)
        correction = 8.0 / torch.sum(self.pairwise_dists_mask,
                                     axis=1)  # correction for some of the dots that have different neighbors
        selected_center_dists[selected_center_dists > 0] = 0
        selected_center_dists[selected_center_dists != 0] = 1

        dtx_centers = 4 * torch.mul(dx_centers,
                                    selected_dxs ** 2 + selected_dys ** 2 - self.pairwise_dists ** 2)
        dty_centers = 4 * torch.mul(dy_centers,
                                    selected_dxs ** 2 + selected_dys ** 2 - self.pairwise_dists ** 2)
        center_dx_update = torch.mul(correction, torch.mul(update_or_not,
                                                           torch.sum(dtx_centers,
                                                                     axis=1)))  # only move if there are events
        center_dy_update = torch.mul(correction, torch.mul(update_or_not, torch.sum(dty_centers, axis=1)))

        # perform the gradient-based update
        calib_center[:, 1] = calib_center[:, 1] - 200 * 0.000015 * torch.clamp(normal_dx_update, -400,
                                                                                         400) + self.dot_regularization_factor * 0.00000025 * center_dx_update
        calib_center[:, 0] = calib_center[:, 0] - 200 * 0.000015 * torch.clamp(normal_dy_update, -400,
                                                                                         400) + self.dot_regularization_factor * 0.00000025 * center_dy_update

        return calib_center
