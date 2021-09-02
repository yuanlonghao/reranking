"""
Publish steps:
- Confirm version number in `setup.py`.
- Delete the old version files in `/dist`.
- Wrap package: `python setup.py sdist bdist_wheel`
- Upload: `twine upload --repository-url https://upload.pypi.org/legacy/ dist/*` (username&password required)
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reranking",
    version="0.2.2",
    author="Longhao Yuan",
    author_email="yuanlonghao1013@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yuanlonghao/reranking",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)
