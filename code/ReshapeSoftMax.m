classdef ReshapeSoftMax < dagnn.ElementWise
  methods
    function outputs = forward(self, inputs, params)
        %liu@0811: added reshape functionality
        [h, w, ch] = size(inputs{1});
        assert(ch == 14, 'the 3rd dimension of cls score must be 14');
        input_reshaped = reshape(inputs{1}, [h w*7 2]);
        %outputs{1} = vl_nnsoftmax(inputs{1}) ;
        outputs{1} = vl_nnsoftmax(input_reshaped);
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
        %liu@0811: added reshape functionality
        [h, w, ch] = size(inputs{1});
        assert(ch == 14, 'the 3rd dimension of cls score must be 14');
        input_reshaped = reshape(inputs{1}, [h w*7 2]);
        
        %derInputs{1} = vl_nnsoftmax(inputs{1}, derOutputs{1}) ;
        derInputs_reshape = vl_nnsoftmax(input_reshaped, derOutputs{1});
        derInputs{1} = reshape(derInputs_reshape, [h w 14]);
        derParams = {} ;
    end

    function obj = ReshapeSoftMax(varargin)
      obj.load(varargin) ;
    end
  end
end