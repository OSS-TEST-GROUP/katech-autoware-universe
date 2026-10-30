from glob import glob
from setuptools import setup

name = 'autoware_korean_traffic_light_models'
setup(
    name=name, version='0.1.0', packages=[name],
    package_data={name: ['manifest.json']},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + name]),
        ('share/' + name, ['package.xml', 'README.md', 'requirements-onnx.txt']),
        ('share/' + name + '/launch', glob('launch/*.launch.py')),
        ('share/' + name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'], zip_safe=False,
    maintainer='JunSongKatech', maintainer_email='songj@katech.re.kr',
    description='Opt-in Korean traffic-light model candidate', license='Apache-2.0',
    entry_points={'console_scripts': [
        'models = ' + name + '.model_io:main',
        'classifier_shadow = ' + name + '.shadow_node:main',
    ]},
)
