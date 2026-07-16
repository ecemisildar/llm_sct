import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ecem/sct_ws/src/llm_sct/install/leo_image_processing'
