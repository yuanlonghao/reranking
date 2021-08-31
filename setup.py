import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reranking",
    version="0.1.0",
    author="Longhao Yuan",
    author_email="yuanlonghao1013@gmail.com",
    long_description=long_description,
    long_distribution_content_type="text/markdown",
    url="https://github.com/yuanlonghao/reranking",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)
