import setuptools

setuptools.setup(
    name='dune',
    version='0.1',
    author='Jay Baptista',
    author_email='jay.baptista@yale.edu',
    description='A small example Python package',
    packages=setuptools.find_packages(include=['dune', 'dune.*']),
    python_requires='>=3',
    install_requires=['numpy>1.7', 'scipy']
)
