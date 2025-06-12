import setuptools
import yaml

# read the contents of requirements.txt
# with open('./requirements.txt') as f:
#     requirements = f.read().splitlines()

def get_requirements_from_env_yml():
    with open('./environment.yml', 'r') as f:
        env_data = yaml.safe_load(f)
    
    requirements = []
    
    # Get pip dependencies if they exist
    dependencies = env_data.get('dependencies', [])
    for dep in dependencies:
        if isinstance(dep, dict) and 'pip' in dep:
            requirements.extend(dep['pip'])
    
    return requirements

requirements = get_requirements_from_env_yml()

setuptools.setup(
    name = 'adrd',
    version = '0.0.1',
    author = 'Sahana Kowhsik',
    author_email = 'skowshik@bu.edu',
    url = 'https://github.com/vkola-lab/ncomms2025/',
    # description = '',
    packages = setuptools.find_packages(),
    python_requires = '>=3.11',
    classifiers = [
        'Environment :: GPU :: NVIDIA CUDA',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    install_requires = requirements,
)