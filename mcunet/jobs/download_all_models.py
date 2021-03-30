import os

if __name__ == '__main__':
    config_list = os.listdir('assets/configs')
    config_list = [c for c in config_list if c != 'proxyless_mobile.json']
    for cfg in config_list:
        pth_path = 'https://hanlab.mit.edu/projects/tinyml/mcunet/release/{}'.format(cfg.replace('.json', '.pth'))
        os.system('wget -P assets/pt_ckpt/ {}'.format(pth_path))
        tflite_path = 'https://hanlab.mit.edu/projects/tinyml/mcunet/release/{}'.format(cfg.replace('.json', '.tflite'))
        os.system('wget -P assets/tflite/ {}'.format(tflite_path))
    print('Done.')
