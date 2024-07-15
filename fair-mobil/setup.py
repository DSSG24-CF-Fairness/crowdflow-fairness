from setuptools import setup

setup(
    name='fmb',
    version='0.0.4',    
    description='A package providing models and analysis for fair mobility',
    url='https://github.com/zukixa/fmb',
    author='zukixa',
    author_email='56563509+zukixa@users.noreply.github.com',
    license='MIT',
    packages=['fmb'],
    entry_points={
        'console_scripts': [
            'fmb=fmb.cli:main',
        ],
    },
    install_requires=['scikit_mobility>=1.3.1',
                      'numpy==1.26.4',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
)