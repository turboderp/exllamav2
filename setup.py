from setuptools import setup

version = "0.0.2"

setup(
    name = "exllamav2",
    version = version,
    packages = ["exllamav2", "exllamav2.generator"],
    url = "https://github.com/turboderp/exllamav2",
    license = "AGPL",
    author = "bb",
    install_requires = [
        "pandas",
        "ninja",
        "fastparquet",
        "torch>=2.0.1",
        "safetensors>=0.3.2",
        "sentencepiece>=0.1.97",
    ],
    include_package_data = True,
)
