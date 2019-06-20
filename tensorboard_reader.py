def launchTensorBoard():
    import os
    PATH = os.getcwd()
    LOG_DIR = PATH + '/LOGS/'
    os.system('tensorboard --logdir=' + LOG_DIR)
    return

import threading
t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()