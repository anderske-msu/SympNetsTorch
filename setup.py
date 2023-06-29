import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SympNetTorch",
    version="0.0.1",
    author="Kelly Anderson",
    author_email="anderske@frib.msu.edu",
    description="Sympletic PyTorch Modules",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anderske-msu/SympNetsTorch",
    project_urls={
        "Bug Tracker": "https://github.com/anderske-msu/SympNetsTorch/issues"
    },
    packages=["SympNetTorch"],
    install_requires=["torch", "sklearn"],
)
