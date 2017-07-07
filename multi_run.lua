-- require 'posix'
require 'lfs'
require 'xlua'
--local indices = {0.4, 0.2, 1, 0.7, 0.01, 0.02, 0.05, 0.1}
local indices = {1}

--local data_classes = {'one'}
local dataset = 'cifar10' 
local data_folder = '../streams/data/' .. dataset .. '/original_data/png/train/'
local data_classes = {}
data_classes['mnist'] = {'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'one', 'zero'}
data_classes['cifar10'] = {'automobile', 'dog', 'cat', 'bird', 'airplane', 'truck', 'deer', 'frog', 'horse', 'ship'}
--data_classes['cifar10'] = {'cat'}

for idx_ratio = 1, table.getn(indices) do
  for idx_class = 1, table.getn(data_classes[dataset]) do
--[[    posix.stdlib.setenv('RATIO_', indices[idx_ratio]) 
    posix.stdlib.setenv('CLASS_', data_classes[idx_class])  ]]  
    os.execute('rm -f myimages/images/*') 
    os.execute('rm -f cache/*')
    nb_files = 0 
    files_list = {}
    for file_ in lfs.dir(data_folder) do
      if string.match(file_, data_classes[dataset][idx_class]) then 
        nb_files = nb_files + 1 
        files_list[nb_files] = file_
      end
    end
    ids = torch.randperm(table.getn(files_list)):long()
    trainset_size = math.floor(ids:size(1)*indices[idx_ratio])
    for idx_image = 1, trainset_size do
      xlua.progress(idx_image, trainset_size)
      os.execute('cp ' .. data_folder .. files_list[ids[idx_image]] .. ' myimages/images/.')
    end
    os.execute('DATA_ROOT=myimages dataset=folder th main.lua')
--     dofile('./main.lua')
    os.execute('cp G_model.t7 ' .. 'models/' .. data_classes[dataset][idx_class] .. '_' .. indices[idx_ratio] .. '.t7')
  end
end



