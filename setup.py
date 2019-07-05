import setuptools


setuptools.setup(
    name = "pcn",
    version = "0.1.0",
    author = "akinori osamura",
    description = "PCN pytorch",
    url = "https://github.com/siriusdemon/pytorch-pcn",
    package = setuptools.find_packages(),
    package_data = {
        'pcn': ['pth/*.pth']
    }
)