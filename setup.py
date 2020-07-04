import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.1.0'
PACKAGE_NAME = 'action_detection'
AUTHOR = 'Kirill Zaitsev'
AUTHOR_EMAIL = 'cyberpank317@gmail.com'
URL = 'https://github.com/cyberpunk317/Action_detection'

LICENSE = 'MIT License'
DESCRIPTION = 'Tool to detect actions in the video stream'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )
