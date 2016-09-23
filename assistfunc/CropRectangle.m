function imgcrop = CropRectangle(img, col, row, width, height)

[h,w,c] = size(img);

% check border
if col < 1 || col + width - 1 > w || row < 1 || row + height - 1 > h
    %warning('MATLAB:ErrorSize', 'Error size of rectangle.');
    col = max(col,1);
    row = max(row,1);
    width = min(width, w - col + 1);
    height = min(height, h - row + 1);
end
% single channel should be converted to RGB channel
if c == 1
    img = repmat(img, [1,1,3]);
end
% 0224 added to avoid invalid point annotation: width == 1 && height == 1 
if width > 1 && height > 1 
    imgcrop = img(row : row + height - 1, col : col + width - 1, :);
else
    imgcrop = []; % otherwise output empty array
end