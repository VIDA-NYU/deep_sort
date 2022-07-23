import setuptools

setuptools.setup(
    name='deep_sort',
    version='0.1.0',
    description='Simple Online Realtime Tracking with a Deep Association Metric',
    long_description=open('README.md').read().strip(),
    long_description_content_type='text/markdown',
    author='Nicolai Wojke',
    url=f'https://github.com/nwojke/deep_sort',
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'opencv-python', 'scipy'],
    license='GNU GENERAL PUBLIC LICENSE',
    keywords='deep sort online realtime tracking object detection tracker')
