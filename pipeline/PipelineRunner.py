import os
import skimage.io as io
import logging
import configparser


class PipelineRunner(object):

    def __init__(self,root_dir, conf_file):
        self.stages_arr = []
        self.root_dir = root_dir

        #init logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        # create a file handler
        log_name = os.path.join(root_dir,'PipelineRunner.log')
        handler = logging.FileHandler(log_name)
        handler.setLevel(logging.DEBUG)
        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(handler)

        #load config
        self.config = configparser.RawConfigParser()

        self.logger.info('Loading configuration file %s',conf_file)
        self.config.read(conf_file)

    def add_stage(self,stage):
        stage.set_config(self.config)
        self.stages_arr.append(stage)

    def execute(self):
        for S in self.stages_arr:
            self.logger.info('Starting stage: %s.',S.get_stage_name())
            nErrors = S.run_stage()
            if nErrors == 0:
                self.logger.info('%s finished without errors.',S.get_stage_name())
            else:
                self.logger.info('%s finished WITH errors.', S.get_stage_name())


    def get_stages(self):
        return self.stages_arr

# def main():
#     if len(sys.argv) != 2:
#         print('Usage: ImageTiler.py <root_dir>')
#         exit()
#
#
# if __name__ == '__main__':
#     main()
