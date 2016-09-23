function rects = PostProcRects(candi_rects)

numCandidates = length(candi_rects);
predicate = eye(numCandidates); % i and j belong to the same group if predicate(i,j) = 1
overlappingThreshold = 0.5;
% mark nearby detections
for i = 1 : numCandidates
    for j = i + 1 : numCandidates
        h = min(candi_rects(i).row + candi_rects(i).size, candi_rects(j).row + candi_rects(j).size) - max(candi_rects(i).row, candi_rects(j).row);
        w = min(candi_rects(i).col + candi_rects(i).size, candi_rects(j).col + candi_rects(j).size) - max(candi_rects(i).col, candi_rects(j).col);
        s = max(h,0) * max(w,0);
        
        if s / (candi_rects(i).size^2 + candi_rects(j).size^2 - s) >= overlappingThreshold
            predicate(i,j) = 1;
            predicate(j,i) = 1;
        end
    end
end

% merge nearby detections
[label, numCandidates] = Partition(predicate);

rects = struct('row', zeros(numCandidates,1), 'col', zeros(numCandidates,1), ...
    'size', zeros(numCandidates,1), 'score', zeros(numCandidates,1), ...
    'neighbors', zeros(numCandidates,1));

for i = 1 : numCandidates
    index = find(label == i);
    %weight = Logistic([candi_rects(index).score]');
    weight = [candi_rects(index).score]';
    rects(i).score = sum( weight );
    rects(i).neighbors = length(index);
    
    if sum(weight) == 0
        weight = ones(length(weight), 1) / length(weight);
    else
        weight = weight / sum(weight);
    end

    rects(i).size = floor([candi_rects(index).size] * weight);
    rects(i).col = floor(([candi_rects(index).col] + [candi_rects(index).size]/2) * weight - rects(i).size/2);
    rects(i).row = floor(([candi_rects(index).row] + [candi_rects(index).size]/2) * weight - rects(i).size/2);
end

clear candi_rects;

% find embeded rectangles
predicate = false(numCandidates); % rect j contains rect i if predicate(i,j) = 1

for i = 1 : numCandidates
    for j = i + 1 : numCandidates
        h = min(rects(i).row + rects(i).size, rects(j).row + rects(j).size) - max(rects(i).row, rects(j).row);
        w = min(rects(i).col + rects(i).size, rects(j).col + rects(j).size) - max(rects(i).col, rects(j).col);
        s = max(h,0) * max(w,0);

        if s / rects(i).size^2 >= overlappingThreshold || s / rects(j).size^2 >= overlappingThreshold
            predicate(i,j) = true;
            predicate(j,i) = true;
        end
    end
end

flag = true(numCandidates,1);

% merge embeded rectangles
for i = 1 : numCandidates
    index = find(predicate(:,i));

    if isempty(index)
        continue;
    end

    s = max([rects(index).score]);
    if s > rects(i).score
        flag(i) = false;
    end
end

rects = rects(flag);

% check borders
% [height, width, ~] = size(I);
% numFaces = length(rects);
% 
% for i = 1 : numFaces
%     if rects(i).row < 1
%         rects(i).row = 1;
%     end
%     
%     if rects(i).col < 1
%         rects(i).col = 1;
%     end
%     
%     rects(i).height = rects(i).size;
%     rects(i).width = rects(i).size;
%     
%     if rects(i).row + rects(i).height - 1 > height
%         rects(i).height = height - rects(i).row + 1;
%     end
%     
%     if rects(i).col + rects(i).width - 1 > width
%         rects(i).width = width - rects(i).col + 1;
%     end    
% end
end


function Y = Logistic(X)
    Y = log(1 + exp(X));
    Y(isinf(Y)) = X(isinf(Y));
end