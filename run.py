import sys, subprocess
import os.path as osp


if __name__ == '__main__':
    models = ['gan', 'dcgan', 'wgan']
    cmd = sys.argv[1]
    assert cmd in models, 'Invalid command'

    runner = sys.executable if sys.executable else 'python'
    runfile = osp.join(osp.abspath(osp.dirname(__file__)), f'{cmd}.py')
    subprocess.check_call([runner, runfile] + sys.argv[2:])
