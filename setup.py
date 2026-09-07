from setuptools import find_packages, setup
from typing import List

def get_requiements(file_path: str) -> List[str]:
    ''' This function will return the list of requirements '''
    Hypen_dot = "-e ."
    requirements = []
    with open(file_path) as file_obj:
        requirments = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
    
        if Hypen_dot in requiements:
            requirements.remove(Hypen_dot)

    return requirements
        

setup(
    name= "Machine Learning Project",
    version = "0.1.0",
    author = "Monodip Das",
    author_email = "dasmonodip108@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)