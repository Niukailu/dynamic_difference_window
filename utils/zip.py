''' zip help module '''
import os


def zip(floder):
    ''' zip floder '''
    assert os.system(f'7za a -r {floder}.7z {floder} > /dev/null') == 0, f'7za a -r {floder}.7z {floder} > /dev/null'
    assert os.system(f'rm {floder} -r') == 0, f'rm {floder} -r'


def unzip(in_floder, out_floder='./'):
    ''' unzip floder '''
    assert os.system(f'7za x {in_floder}.7z -r -o{out_floder} > /dev/null') == 0, f'7za x {in_floder}.7z -r -o{out_floder} > /dev/null'
