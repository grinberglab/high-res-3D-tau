function new_img = histogram_match(img_name_ref,img_name_tgt)

%REF : reference image
%TGT: image that needs to be normalized

    img_ref= niftiread(img_name_ref);
    img_tgt = niftiread(img_name_tgt);

    img_ref = img_ref./1000;
    img_tgt = img_tgt./1000;
    [rows cols N] = size(img_tgt);

    x_ref = img_ref(img_ref > 0.01);
    x_tgt = img_tgt(img_tgt > 0.01); % more than 1% of tau inside 1^2 um of tissue

    h_ref = histogram(x_ref,1000);figure,
    h_tgt = histogram(x_tgt,1000);

    h_rnorm = h_ref.Values/sum(h_ref.Values);
    h_tnorm = h_tgt.Values/sum(h_tgt.Values);

    cdf_ref = cumsum(h_rnorm);
    cdf_tgt = cumsum(h_tnorm);
    
    %create bin edges matrix  
    lenBinEdges = length(h_tgt.BinEdges);
    bin_edges_tgt = zeros(3,lenBinEdges-1);
    for i = 1:lenBinEdges-1
        bin_edges_tgt(1,i) = h_tgt.BinEdges(i);
        bin_edges_tgt(2,i) = h_tgt.BinEdges(i+1);
        bin_edges_tgt(3,i) = i;
    end 
    lenBinEdges = length(h_ref.BinEdges);
    bin_edges_ref = zeros(3,lenBinEdges-1);
    for i = 1:lenBinEdges-1
        bin_edges_ref(1,i) = h_ref.BinEdges(i);
        bin_edges_ref(2,i) = h_ref.BinEdges(i+1);
        bin_edges_ref(3,i) = i;
    end 
    
    %compute new heatmap value, only process values >= 0.01 
    new_img = zeros(rows,cols);
    idx_values = find(img_tgt >= 0.01);
    nVal = length(idx_values);
    for i=1:nVal
        idx = idx_values(i);
        v = img_tgt(idx);
        %find bin in target histogram
        [bin_idx_tgt, tgt_range] = find_bin(bin_edges_tgt,v);
        v_cdf_tgt = cdf_tgt(bin_idx_tgt);           
        %find closest match to tgt cdf value in ref cdf
        [d,idx_cdf_ref] = min(abs(cdf_ref - v_cdf_tgt));
        ref_range = bin_edges_ref(:,idx_cdf_ref); 
        %compute new value
        new_v = linear_map(tgt_range,ref_range,v);
        new_img(idx) = new_v;
    end     
    
    new_img = new_img*1000;

end

function new_value = linear_map(tgt,ref,x)
    new_value = (((ref(2)-ref(1))*(x - tgt(1))) / (tgt(2) - tgt(1))) + ref(1);
end

%
% find bin by binary search
%
function [bin_idx, tgt_range] = find_bin(bin_edges, v)

    bin_idx = -1;
    tgt_range = -1;
    
    %bin_edges has to be transformed to a [2 3] vector with row1 = bin
    %start, row2 = bin end
    
    [rows cols N] = size(bin_edges);
    nBins = cols;
    mid_idx = round(nBins/2);
    
    %is value inside middle bin? if yes, found!
    if v >= bin_edges(1,mid_idx) && v <= bin_edges(2,mid_idx)
        bin_idx = bin_edges(3,mid_idx); %global bin index 
        tgt_range = [bin_edges(1,mid_idx) bin_edges(2,mid_idx)];
        return
    else
        if v < bin_edges(1,mid_idx)
            bin_edges_new = bin_edges(:,1:mid_idx-1);
            [bin_idx,tgt_range] = find_bin(bin_edges_new,v);
        elseif v > bin_edges(2,mid_idx)
            bin_edges_new = bin_edges(:,mid_idx+1:end);
            [bin_idx,tgt_range] = find_bin(bin_edges_new,v);
        end
    end    
end




