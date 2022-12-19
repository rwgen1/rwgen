from setuptools import setup

setup(
    name='rwgen',
    version='0.0.3',
    packages=['rwgen', 'rwgen.tests', 'rwgen.weather', 'rwgen.rainfall'],
    url='https://github.com/davidpritchard1/rwgen',
    license='GPL-3.0 license',
    author='ndp81',
    author_email='',
    description='stochastic spatiotemporal Rainfall and Weather GENerator',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
    ],
)
