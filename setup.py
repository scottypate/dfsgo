from setuptools import setup, find_packages
setup(
    name="dfsgo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.13.3', 'pandas>=0.20.3'],
    author="Scotty Pate",
    author_email="scottypate@me.com",
    description="Genetic search optimizer for daily fantasy lineups",
    url="http://github.com/scottypate/dfsgo.git"
)