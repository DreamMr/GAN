from .BaseOptions import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self,parser):
        parser = BaseOptions.initialize(self,parser)
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument("--test_out_name",type=str,default='./test_out/cgan/origin/')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--out_number', type=int, default=600, help='input batch size')

        self.is_train = False
        return parser