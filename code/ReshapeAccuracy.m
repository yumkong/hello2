classdef ReshapeAccuracy < dagnn.ElementWise
  properties
        anchorNum = 7 %5 -- test, 9 -- train
        loss = 'classerror'
        opts = {}
  end
  
  properties (Transient)
        average = 0
        numAveraged = 0
  end
    
  methods
    function outputs = forward(self, inputs, params)
        %liu@0811: added reshape functionality
        %liu@0811: added reshape functionality
        [w_, h_, ~] = size(inputs{1});
        input_reshaped = reshape(inputs{1}, [w_ h_*self.anchorNum 2]);
        %outputs{1} = vl_nnsoftmax(input_reshaped);
        
        [~, predlabel] = max(input_reshaped,[],3);
       
        c = inputs{2}; %label
        label = c(:,:,1,:);
        samelabel = single(predlabel ~= label);
        outputs{1} = sum(samelabel(:)) / numel(label);
        
        %liu@0812: 
        n = self.numAveraged ;
        m = n + size(inputs{1},4) ;
        self.average = (n * self.average + double(gather(outputs{1}))) / m ;
        self.numAveraged = m;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
        %liu@0811: added reshape functionality
        x = inputs{1};
        if isa(x,'gpuArray')
            y = gpuArray.zeros(size(x),classUnderlying(x)) ;
        else
            y = zeros(size(x),'like',x) ;
        end
        derInputs{1} = y;
        derInputs{2} = [];
        derParams = {} ;
    end

    function obj = ReshapeAccuracy(varargin)
      obj.load(varargin) ;
    end
  end
end