function net = rpnInitializeModel(varargin)
%RPNINITIALIZEMODEL Initialize the faster RCNN region proposal model from VGG-VD-16

opts.sourceModelUrl = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat' ;
opts.sourceModelPath = 'data/models/imagenet-vgg-verydeep-16.mat' ;
% receive the parameters from outside
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                    Load & download the source model if needed (VGG VD 16)
% -------------------------------------------------------------------------
if ~exist(opts.sourceModelPath, 'file')
    fprintf('%s: downloading %s\n', opts.sourceModelUrl) ;
    mkdir(fileparts(opts.sourceModelPath)) ;
    urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat', opts.sourceModelPath) ;
end
net = vl_simplenn_tidy(load(opts.sourceModelPath)) ;

% for convt (deconv) layers, cuDNN seems to be slower?
net.meta.cudnnOpts = {'cudnnworkspacelimit', 512 * 1024^3} ;

%only keep layers from conv1_1 to relu5_3, remove fully connected layers
net.layers = net.layers(1:30);

% Convert the model from SimpleNN to DagNN
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

% conv1_1 w rgb --> bgr
net.params(1).value = net.params(1).value(:,:,[3,2,1],:);
for j = 1:2:26
    net.params(j).value = permute(net.params(j).value, [2,1,3,4]);
end
% stablize the learning rates of conv1_1, conv1_2, conv2_1 and conv2_2
for j = 1:8
   net.params(j).learningRate = 0;
end
% from conv 3_1 to conv5_3, b's learning rate is set to 2 (previous: 1) 
for j = 10:2:26
   net.params(j).learningRate = 2;
end

% -------------------------------------------------------------------------
% add new layers after conv5_3
% -------------------------------------------------------------------------
% new layer1: 3x3 conv layer
% % ============= start random ======================
% rng_seed = 6;
% prev_rng = rng;  %keep previous random state
% rng(rng_seed, 'twister');

% ============== load same parameters as in caffe ====================
%init_param{1} -- conv_prop1f; init_param{2} -- conv_prop1b
%init_param{3} -- prop_cls1f; init_param{4} -- prop_cls1b
%init_param{5} -- prop_bbox1f; init_param{6} -- prop_bbox1b
load('param_cell.mat');
init_param = param_cell;
clear param_cell
% ==================================
net.addLayer('conv_proposal1', ...
              dagnn.Conv('size', [3 3 512 512],...
                          'hasBias', true, ...
                          'stride', [1, 1],...
                          'pad', [1 1 1 1]), ...
              {'x30'}, {'conv_proposal1'}, ...  %{input layer} + {output layer(generally same as this layer's name)}
              {'conv_prop1f'  'conv_prop1b'}) ; % parameters of this layer
f_idx = net.getParamIndex('conv_prop1f') ;
% liu@0908 changed to be the same as caffe
%net.params(f_idx).value = 0.01*randn(3,3,512,512, 'single'); %gaussian with mean 0, std 0.01
net.params(f_idx).value = init_param{1}; %permute(init_param{1}, [2,1,3,4]); %gaussian with mean 0, std 0.01
net.params(f_idx).learningRate = 1;
net.params(f_idx).weightDecay = 1;
b_idx = net.getParamIndex('conv_prop1b') ;
% liu@0908 changed to be the same as caffe
%net.params(b_idx).value = zeros(1,512, 'single'); %vector with all elements 0
net.params(b_idx).value = init_param{2};
net.params(b_idx).learningRate = 2;
net.params(b_idx).weightDecay = 1;

% new layer2: relu for new layer1
net.addLayer('relu_proposal1', dagnn.ReLU(), ...
            {'conv_proposal1'}, {'relu_proposal1'}, {});  %{input layer} + {output layer} + {param} 
        
% new layer3: 1x1 conv layer for proposal classfication
net.addLayer('proposal_cls_score', ...
              dagnn.Conv('size', [1 1 512 14],...  % be be changed to 2(face/nonface) * 9(#anchors)
                          'hasBias', true, ...
                          'stride', [1, 1],...
                          'pad', [0 0 0 0]), ...
              {'relu_proposal1'}, {'proposal_cls_score'}, ...  %{input layer} + {output layer(generally same as this layer's name)}
              {'prop_cls1f'  'prop_cls1b'}) ; % parameters of this layer
f_idx = net.getParamIndex('prop_cls1f') ;
% liu@0908 changed to be the same as caffe
%net.params(f_idx).value = 0.01*randn(1,1,512,14, 'single'); %gaussian with mean 0, std 0.01
net.params(f_idx).value = init_param{3}; %permute(init_param{3}, [2,1,3,4]); %gaussian with mean 0, std 0.01
net.params(f_idx).learningRate = 1;
net.params(f_idx).weightDecay = 1;
b_idx = net.getParamIndex('prop_cls1b') ;
% liu@0908 changed to be the same as caffe
%net.params(b_idx).value = zeros(1,14, 'single'); %vector with all elements 0
net.params(b_idx).value = init_param{4};
net.params(b_idx).learningRate = 2;  %0907 changed
net.params(b_idx).weightDecay = 1;

% new layer4: 1x1 conv layer for proposal bbox regress
net.addLayer('proposal_bbox_pred', ...
              dagnn.Conv('size', [1 1 512 28],...  % be be changed to 4(bbox coord targets) * 9(#anchors)
                          'hasBias', true, ...
                          'stride', [1, 1],...
                          'pad', [0 0 0 0]), ...
              {'relu_proposal1'}, {'proposal_bbox_pred'}, ...  %{input layer} + {output layer(generally same as this layer's name)}
              {'prop_bbox1f'  'prop_bbox1b'}) ; % parameters of this layer
f_idx = net.getParamIndex('prop_bbox1f') ;
% liu@0908 changed to be the same as caffe
%net.params(f_idx).value = 0.01*randn(1,1,512,28, 'single'); %gaussian with mean 0, std 0.01
net.params(f_idx).value = init_param{5}; %permute(init_param{5}, [2,1,3,4]);
net.params(f_idx).learningRate = 1;
net.params(f_idx).weightDecay = 1;
b_idx = net.getParamIndex('prop_bbox1b') ;
% liu@0908 changed to be the same as caffe
%net.params(b_idx).value = zeros(1, 28, 'single'); %vector with all elements 0
net.params(b_idx).value = init_param{6};
net.params(b_idx).learningRate = 2;
net.params(b_idx).weightDecay = 1;

% new layer5: softmax loss for classification
%net.addLayer('prediction', ReshapeSoftMax(), {'proposal_cls_score'}, {'prediction'}, {});  % dagnn.SoftMax() --> ReshapeSoftMax
%net.addLayer('loss_cls', WeightLoss('loss', 'log'), {'prediction', 'label',}, {'loss_cls'}, {});
net.addLayer('loss_cls', ReshapeSoftmaxLoss('loss', 'softmaxlog'), {'proposal_cls_score', 'label',}, {'loss_cls'}, {});

% new layer6: accuracy for classification
%net.addLayer('accuracy', WeightLoss('loss', 'classerror'), {'prediction','label'}, 'accuracy') ;  % --> ReshapeLoss
net.addLayer('accuracy', ReshapeAccuracy('loss', 'classerror'), {'proposal_cls_score','label'}, 'accuracy') ;  % --> ReshapeLoss

% new layer7: loss for bbox regression
net.addLayer('loss_bbox', SmoothL1Loss('loss', 'smoothl1'), {'proposal_bbox_pred','bbox_targets', 'bbox_loss'}, 'loss_bbox') ;

% keep the following intermediate values
net.vars(34).precious = true;  % cls_score
net.vars(36).precious = true;  % label
net.vars(37).precious = true;  % loss_cls
net.vars(41).precious = true;  % loss_bbox