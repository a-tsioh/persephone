from setuptools import setup

setup(name='persephone',
<<<<<<< HEAD
      version='0.1.7',
      description='A tool for automatic phoneme transcription',
=======
      version='0.1.8',
      description='A tool for developing automatic phoneme transcription models',
>>>>>>> master
      long_description=open('README.md').read(),
      url='https://github.com/oadams/persephone',
      author='Oliver Adams',
      author_email='oliver.adams@gmail.com',
      license='GPLv3',
      packages=['persephone', 'persephone.datasets'],
      classifiers = [
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
      ],
      keywords='speech-recognition machine-learning acoustic-models artificial-intelligence neural-networks',
      install_requires=[
           'ipython==6.2.1',
           'GitPython==2.1.8',
           'nltk==3.2.5',
           'numpy==1.14.0',
           'python-speech-features==0.6',
           'scipy==1.0.0',
           'tensorflow==1.4.1',
           'scikit-learn==0.19.1',
           'pympi-ling==1.69',
      ],
)
