function launch_net(conf, imdb, net_handle, experimentID, opt_GPUID, use_conv4, preNet, varargin)
	
    % initialize GPU
    GPU_ID = opt_GPUID;
    if(~isempty(opt_GPUID))
        gpuDevice(GPU_ID);
    end

    % create experiment directory and save configuration
    if(~exist(experimentID, 'dir'))
        mkdir(experimentID);
    end

    % directory to save the result
    input_opts.expDir = experimentID;
    % 20160510 added
    input_opts.gpus = GPU_ID;
    %liu:0921 added
    input_opts.use_conv4 = use_conv4;

    % call the net to start the experiment
    % liu@0808: changed
    %net_handle =  str2func(netName);
    [fnet, infor] = net_handle(conf, imdb, preNet, input_opts, varargin);

    % save net and info
    save([experimentID '/fnet.mat'], 'fnet');
    save([experimentID '/infor.mat'], 'infor');
end
