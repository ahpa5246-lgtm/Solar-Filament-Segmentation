"""Build the private native-resolution pilot."""
from build_v6_refiner_notebook import build
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    build(p.parse_args().output,experiment='confirmation')
