dofile('../../streams/data/data_preparation/generate_data_from_models.lua')
dofile('../../streams/data/data_preparation/visualize_data.lua')
require 'cudnn'
opt = {}
opt.dataset = 'other'
opt.gen_per_class = 100
opt.models_folder = '../../streams/models/pretrained_generative_models/mnist_by_train_size/s07/'
data = generate_from_models_set(opt)
show_multiple_images(data, 10, 10)

function generate_from_all_folders()
  local root = '/home/abesedin/workspace/Projects/streams/models/pretrained_generative_models/mnist_by_train_size/'
  local folders = {'s001/', 's002/', 's005/', 's01/', 's02/', 's04/', 's07/', 's1/'}
  local models = {'zero.t7', 'one.t7', 'two.t7', 'three.t7', 'four.t7', 'five.t7', 'six.t7', 'seven.t7', 'eight.t7', 'nine.t7'}
  n1 = table.getn(folders)
  n2 = table.getn(models)
  local out_image = torch.zeros((n1+1)*32+2, n2*32)
  for idx1 = 1, n1 do
    for idx2 = 1, n2 do
      local in_noise = torch.randn(10,100, 1, 1):normal():cuda()
      local model_path = root .. folders[idx1] ..  models[idx2]
      local model = torch.load(model_path)
      local out = model:forward(in_noise):float()
      res = nn.SpatialAveragePooling(2,2,2,2):float():forward(out)
      res = (res - res:mean())/res:std()
      out_image[{{1 + (idx1-1)*res:size(3),idx1*res:size(3)},{1 + (idx2-1)*res:size(4),idx2*res:size(4)}}] = res[{{1},{1},{},{}}]:squeeze()
    end
  end
  out_image[{{1 + n1*res:size(3), 2 + n1*res:size(3)},{}}]:fill(1)
  local orig_data = torch.load('/home/abesedin/workspace/Projects/streams/data/mnist/original_data/t7/train.t7', 'ascii')
  orig_data.data = orig_data.data:float()
  local data = orig_data.data/255*2 - 1
  cur_idx = 1
  for idx = 1, 10 do
    while true do
      if orig_data.labels[cur_idx] == idx then
        data[{{cur_idx},{1},{},{}}] = (data[{{cur_idx},{1},{},{}}] - data[{{cur_idx},{1},{},{}}]:mean())/data[{{cur_idx},{1},{},{}}]:std()
        out_image[{{3 + n1*res:size(3), 2 + (n1 + 1)*res:size(3)},{1 + (idx-1)*res:size(4),idx*res:size(4)}}] = data[{{cur_idx},{1},{},{}}]:squeeze()
        cur_idx = cur_idx + 1
        break
      else
        cur_idx = cur_idx + 1
      end
    end
  end
  return out_image
end
      