from pprint import pprint
import os
import setproctitle

class Config:
    name = 'roadTec_NewRoadTec_fileChange'

    gpu_id = '0,1,2,3'

    setproctitle.setproctitle("%s" % name)

    classification_path = 'D:/yolov5-master/yolov5-master/data/temp'
    detect_save_path = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/'
    detect_path = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/result/'
    detect_image_save_path = 'D:/DeepCrack-master/DeepCrack-master/codes/results/'
    classification_Data_path = 'D:/sample/deepTest/data'
    # path
    #test_data_path = 'data/test_dataPath.txt'
    #Train val path
    test_data_path = 'data/val_test.txt'
    #Train path
    train_data_path = 'data/train_example.txt'

    #train_data_path = 'data/train_test.txt'
    val_data_path = 'data/test_example.txt'
    checkpoint_path = 'checkpoints'
    log_path = 'log'
    saver_path = os.path.join(checkpoint_path, name)
    
    max_save = 20
    bCheck =False
    iIndex = -1
    
    # visdom
    vis_env = 'DeepCrack'
    port = 8097
    vis_train_loss_every = 40
    vis_train_acc_every = 40
    vis_train_img_every = 120
    val_every = 200

    # training
    epoch = 300
    pretrained_model = ''
    pretrained_model_temp=''
    weight_decay = 0.0000
    lr_decay = 0.1
    lr = 1e-3
    momentum = 0.9
    use_adam = True  # Use Adam optimizer
    train_batch_size = 1
    val_batch_size = 2
    test_batch_size = 2

    acc_sigmoid_th = 0.5
    pos_pixel_weight = 1

    # checkpointer
    save_format = 'hoit'
    save_acc = -1
    save_pos_acc = -1

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

    def show(self):
        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')