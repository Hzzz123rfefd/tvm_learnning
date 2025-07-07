import numpy as np
from tvm_common.common import inference, get_common_parser

def main(args):
    x = np.ones((1, 784)).astype("float32")
    y = inference(args, x)

if __name__ == "__main__":
    parser = get_common_parser()
    args = parser.parse_args()
    main(args)