import subprocess

broken=subprocess.run(['pip', 'check'], capture_output=True, text=True)
with open('broken.txt', 'wb') as f:
     f.write(broken.stdout.encode('utf-8'))

installed = subprocess.run(['pip', 'list'], capture_output=True, text=True)
with open('installed.txt', 'wb') as f:
    f.write(installed.stdout.encode('utf-8'))

requirements = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)
with open('requirements.txt', 'wb') as f:
    f.write(requirements.stdout.encode('utf-8'))

sklearninfo = subprocess.run(['python', '-m', 'pip', 'show', 'scikit-learn'], capture_output=True, text=True)
with open('sklearninfo.txt', 'wb') as f:
    f.write(sklearninfo.stdout.encode('utf-8'))
    


