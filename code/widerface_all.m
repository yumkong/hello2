function dataset = widerface_all(dataset, usage, use_flip)
% Pascal voc 2007 trainval set
% set opts.imdb_train opts.roidb_train 
% or set opts.imdb_test opts.roidb_train

% change to point to your devkit install
% in puck
%devkit = '/usr/local/data/yuguang/dataset/wider_face';
% in VN7_win
devkit = 'D:\\datasets\\WIDERFACE';

switch usage
    case {'train'}
        %dataset.imdb_train    = {  imdb_from_widerface(devkit, 'trainval', use_flip) };
        %dataset.roidb_train   = cellfun(@(x) x.roidb_func(x), dataset.imdb_train, 'UniformOutput', false);
        [dataset.imdb_train, dataset.roidb_train] = imdb_from_widerface(devkit, 'trainval', use_flip);
    case {'test'}
        %dataset.imdb_test     = imdb_from_widerface(devkit, 'test', use_flip) ;
        %dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test);
        [dataset.imdb_test, dataset.roidb_test] = imdb_from_widerface(devkit, 'test', false);
    otherwise
        error('usage = ''train'' or ''test''');
end

end