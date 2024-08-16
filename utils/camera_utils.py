import dv_processing as dv

def get_all_available_cameras():
    # function to return which cameras are connected to this PC
    return dv.io.discoverDevices()

def set_internal_integration_time(camera, time=1000):
    camera.deviceConfigSet(-3, 1, time)

def get_internal_integration_time(camera):
    return camera.deviceConfigGet(-3, 1)