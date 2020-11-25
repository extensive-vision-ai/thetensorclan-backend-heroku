# Development Environment

## Create a Conda Environment

Clean your environment and packages
```shell script
conda clean --all
```

```shell script
conda env create -f devenv/environment.yaml
```

or install the packages manually
```shell script
conda create -n thetensorclan-aws python=3.7
conda install pytorch torchvision cudatoolkit=10.2 torchtext torchaudio -c pytorch
conda install -c anaconda flask boto3 flask-cors
conda install -c conda-forge dlib opencv spacy ffmpeg-python
pip install onnxruntime SoundFile
```

## Notes

exporting environment
```shell script
conda env export --no-builds > devenv/environment.yml
```

use black for formatting code
```shell script
pip install black
```

follow this guide: https://black.readthedocs.io/en/stable/editor_integration.html

