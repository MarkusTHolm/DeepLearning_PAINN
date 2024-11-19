cd 'C:\Users\svida\Documents\GitHubCode\DeepLearning_PAINN\res'

ssh gbar_transfer 'tar --exclude='*.vtu' -C /zhome/19/d/137388/github/DeepLearning_PAINN/runs/train/ -cz ./' | tar -xz