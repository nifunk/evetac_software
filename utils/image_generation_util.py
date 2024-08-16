import torch
import dv_processing as dv
import numpy as np
import os
import onnxruntime as rt

class GenerateImageTorchCuda:
    def __init__(self, camera):
        self.camera = camera
        self.pic_tensor = torch.zeros(camera.getEventResolution(), dtype=torch.int8).cuda()
        self.pic_tensor[:] = 0

    def generate_image(self, events_x, events_y, events_polarity):
        # important: assumes input data in the following form:
        # events_x = torch.from_numpy(events_numpy['x']).cuda().long()
        # events_y = torch.from_numpy(events_numpy['y']).cuda().long()
        # events_polarity = torch.from_numpy(events_numpy['polarity']).cuda().type(torch.int8)

        self.pic_tensor[:] = 0
        if (events_x.shape[0]!=0): # only if there are some events make the rest,...
            xy_idc = torch.vstack((events_x, events_y))
            self.pic_tensor[list(xy_idc)] = self.pic_tensor[list(xy_idc)] + events_polarity

        return self.pic_tensor

    def get_zero_img(self):
        return torch.zeros(self.camera.getEventResolution(), dtype=torch.int8).cuda()


class GenerateImageDvAccumulator:
    # class to generate an image frame using the API provided by the DV module
    def __init__(self, camera):
        self.accumulator = dv.EdgeMapAccumulator(camera.getEventResolution(), 0.0038, False, 0.5)  # increments by 1

    def generate_image(self, event_slice):
        # this function expects an event slice as input, and returns numpy array as int8
        self.accumulator.accept(event_slice)
        frame = self.accumulator.generateFrame()
        img1 = np.transpose((np.asarray(frame.image).astype(np.int16)-127).astype(np.int8))
        self.accumulator.reset()
        return img1

class ImgOnlyOnnx(torch.nn.Module):

    def __init__(self, camera):
        super().__init__()

        self.pic_tensor = torch.zeros(camera.getEventResolution(),dtype=torch.uint8).cpu() + 127
        self.pic_tensor[:] = 127

    def forward(self, events_x, events_y, events_polarity):
        events_x = events_x.long()
        events_y = events_y.long()
        idc1 = [events_x[events_polarity==0], events_y[events_polarity==0]]
        idc2 = [events_x[events_polarity==1], events_y[events_polarity==1]]
        self.pic_tensor[idc1] = 0
        self.pic_tensor[idc2] = 255

        return self.pic_tensor


class ImgOnlyOnnxRelative(torch.nn.Module):

    def __init__(self, camera):
        super().__init__()

    def forward(self, events_x, events_y, events_polarity, pic_tensor):
        events_x = events_x.long()
        events_y = events_y.long()
        idc1 = [events_x[events_polarity==0], events_y[events_polarity==0]]
        idc2 = [events_x[events_polarity==1], events_y[events_polarity==1]]
        pic_tensor[idc1] = pic_tensor[idc1] - 15*1
        pic_tensor[idc2] = pic_tensor[idc2] + 15*1

        return pic_tensor

def export_img_only_net_onnx(camera):
    # export the network to onnx
    net = ImgOnlyOnnxRelative(camera)

    pic_tensor = torch.from_numpy(np.ones((camera.getEventResolution()),dtype=np.uint8))
    events_x = torch.from_numpy(np.zeros((2),dtype=np.int16))
    events_y = torch.from_numpy(np.zeros((2),dtype=np.int16))
    events_polarity = torch.from_numpy(np.zeros((2),dtype=np.int8))

    torch.onnx.export(net,  # model being run
                      (events_x, events_y, events_polarity, pic_tensor), # model input (or a tuple for multiple inputs)
                      os.path.dirname(os.path.realpath(__file__)) + "/onnx_models/" + "img_only.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input1', 'input2', 'input3', 'input4'],  # the model's input names
                      output_names=['output1'],
                      dynamic_axes={'input1': {0: 'batch_size'},  # variable length axes
                                    'input2': {0: 'batch_size'},
                                    'input3': {0: 'batch_size'},
                                    'input4': {0: 'batch_size'}})

class GenerateImageOnnx:
    # class to generate an image frame using the API provided by the DV module
    def __init__(self, camera, gpu=False, generate_model_anew=False):
        self.camera = camera
        self.gpu = gpu
        if (generate_model_anew):
            export_img_only_net_onnx(camera)

        if (gpu):
            self.sess = rt.InferenceSession(os.path.dirname(os.path.realpath(__file__)) + "/onnx_models/" + "img_only.onnx", providers=['CUDAExecutionProvider'])
        else:
            self.sess = rt.InferenceSession(os.path.dirname(os.path.realpath(__file__)) + "/onnx_models/" + "img_only.onnx", providers=['CPUExecutionProvider'])

    def generate_image(self, events_x, events_y, events_polarity, curr_pic):
        # this function if set to use GPU, expects torch tensors on the gpu as input, otherwise np input is fine
        # this function expects an event slice as input, and returns numpy array as int8
        # ort_inputs = {'input1': events_x, 'input2': events_y, 'input3': events_polarity}
        ort_inputs = {'input1': events_x, 'input2': events_y, 'input3': events_polarity, 'input4': curr_pic}
        ort_outs = self.sess.run(None, ort_inputs)
        return ort_outs[0]

    def get_zero_img(self):
        return np.zeros(self.camera.getEventResolution(), dtype=np.uint8) + 127