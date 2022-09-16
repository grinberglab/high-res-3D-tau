function run_histogram_match

%AV2 AT100
%slices = [76, 84, 116, 124, 132, 140, 156, 172, 180, 188, 196, 204, 236, 244, 252, 260, 276, 292, 300, 308, 316, 324, 356, 364, 372, 380, 396, 404, 412, 420, 428, 436, 444, 476, 484, 492, 500, 516, 524, 532, 540, 548, 556, 564, 596, 604, 612, 620, 636, 644, 652, 660, 668, 673, 676, 681, 684, 689, 692, 700, 705, 708, 713, 716, 721, 724, 732, 737, 740, 745, 748, 753, 756, 764];
%slices = [92, 212,332, 452, 572, 100, 220, 340, 460, 580, 228, 348, 468, 588, 148, 268, 388, 508, 628];
%ref_slice_id = 98;

%AV1 AT8
%slices = [282,298,314,330,346,366,386,406,426,448,466,486,506,526,546,566,586,606,626,650,286,302,318,334,350,370,390,410,430,450,470,490,510,530,550,570,590,610,630,654,162,290,306,322,338,354,374,394,414,434,454,474,494,514,534,554,574,594,614,634,658,202,294,310,326,342,358,378,398,418,438,458,478,498,518,538,558,578,598,618,638,702,242,295,311,327,343,362,382,402,422,462,482,502,522,542,562,582,602,622,642];
%slices = [574,594,614,634,658,202,294,310,326,342,358,378,398,418,438,458,478,498,518,538,558,578,598,618,638,702,242,295,311,327,343,362,382,402,422,462,482,502,522,542,562,582,602,622,642];
%ref_slice_id = 442;

%AV2 AT8
slices = [303,307,327,339,351,375,383,411,419,443,463,479,483,591,623,643,647,651,667,691];
ref_slice_id = 259;

dir_prefix = '/home/maryana/storage2/Posdoc/AVID/AV23/AT8/full_res/rescan_AT8_';

ref_name = strcat(dir_prefix,num2str(ref_slice_id),'/heat_map/hm_map_0.1/heat_map_0.1_res10.nii');

nSlices = length(slices);

nError = 0;
for i=1:nSlices
    try
        slice_id = slices(i);

        fprintf('** Slice %d **\n',slice_id);    

        tgt_name = strcat(dir_prefix,num2str(slice_id),'/heat_map/hm_map_0.1/heat_map_0.1_res10.nii');
        new_img_name = strcat(dir_prefix,num2str(slice_id),'/heat_map/hm_map_0.1/heat_map_0.1_norm_res10.nii');

        %tgt_img = niftiread(tgt_name);
        tgt_header = niftiinfo(tgt_name);

        fprintf('Normalizing...\n'); 
        new_img = histogram_match(ref_name,tgt_name);
        fprintf('Saving file %s\n',new_img_name);
        niftiwrite(new_img,new_img_name,tgt_header);
    catch
        fprintf('ERROR! %s\n',new_img_name);
        nError = nError + 1;
    end
end
fprintf('Finished with %d error(s)\n',nError);



