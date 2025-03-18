class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'data':
            return 'D:/zhuomian/deepcrack1/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError