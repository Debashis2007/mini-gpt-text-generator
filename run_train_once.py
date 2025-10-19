import importlib.util
spec = importlib.util.spec_from_file_location('train_gpt','c:/AIML/mini-gpt-text-generator/train_gpt.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('module loaded, train exists?', hasattr(mod,'train'))
if hasattr(mod,'train'):
    print('Calling train(epochs=1)')
    mod.train(epochs=1)
