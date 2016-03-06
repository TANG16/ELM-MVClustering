clear;
clc;
stream = RandStream.getGlobalStream;
reset(stream);
addpath('./graph');
addpath('./code_coregspectral');
tic;
load ./data/SenITVehicle_2views_300samples_3clusters;

num_views = 2;
numClust = 3;
truth = label;
%% Data normalization
data{1} = Acou;
data{2} = Seis;

d1 = mapstd(data{1}');
d2 = mapstd(data{2}');


clear data;
data{1} = d1';
data{2} = d2';
%%==================================================
F_array_avg =[];
P_array_avg =[];
R_array_avg = [];
nmi_array_avg =[];
avgent_array_avg=[];
AR_array_avg =[];
ACC_array_avg=[];
Purity_array_avg=[];


F_array_std =[];
P_array_std =[];
R_array_std = [];
nmi_array_std =[];
avgent_array_std=[];
AR_array_std =[];
ACC_array_std=[];
Purity_array_std=[];
neurons_i = [];

%% ==================ELM Process======================
% nurons ranges from 100 to 20000, with 29 different values
nurons =1000;
[H1,~]=myelm(data{1},truth,nurons,'sigmoid');
[H2,~]=myelm(data{2},truth,nurons,'sigmoid');

X1 =H1;
X2 =H2;

clear data;
data{1}=H1;
data{2}=H2;
%%%%%%%%%%%%%%%%% Step 1: construct graph Laplacian %%%%%%%%%%%%%%%%%
% hyper-parameter settings for graph
options.GraphWeights='binary';
options.GraphDistanceFunction='euclidean';
options.LaplacianNormalize=0;
options.LaplacianDegree=1;
options.NN=5;
L= zeros(size(data{1},1),size(data{1},1),num_views);
for i=1:num_views
    L(:,:,i)=laplacian(options,data{i});
end
max_iter = 50;
step =-2;
pr_array= [];
while step < 2.0001
    for iter =1:max_iter
        % %%%%%%%%%%%%%%%%% Step 2: Clustering %%%%%%%%%%%%%%%%%
        pr = 10^step;
        [G] = MVSpectralClustering(L, numClust,pr, 'kmeans');
        % %%%%%%%%%%%%%%%%% Step 3: Clustering Evaluation%%%%%%%%%%%%%%%%%
        outlabel = zeros(length(label),1);
        for i= 1:numClust
            for j =1:length(label)
                if G(j,i) ==1
                    outlabel(j) = i;%i-1;
                end
            end
        end
        %evaluation 1
        [result,~] = ClusteringMeasure(label, outlabel);
        ACCi(iter) =result(1);
        NMIi(iter) =result(2);
        Purityi(iter)=result(3);
        
        %evaluation2
        [Fi(iter),Pi(iter),Ri(iter)] = compute_f(label,outlabel);
        [A nmii(iter) avgenti(iter)] = compute_nmi(label,outlabel);
        if (min(label)==0)
            [ARi(iter),RIi(iter),MIi(iter),HIi(iter)]=RandIndex(label+1,outlabel+1);
        else
            [ARi(iter),RIi(iter),MIi(iter),HIi(iter)]=RandIndex(label,outlabel);
        end
        
    end
    
    
    F(1) = mean(Fi); F(2) = std(Fi);
    P(1) = mean(Pi); P(2) = std(Pi);
    R(1) = mean(Ri); R(2) = std(Ri);
    nmi(1) = mean(nmii); nmi(2) = std(nmii);
    avgent(1) = mean(avgenti); avgent(2) = std(avgenti);
    AR(1) = mean(ARi); AR(2) = std(ARi);
    
    
    ACC(1) = mean(ACCi); ACC(2) = std(ACCi);
    NMI(1) = mean(NMIi); NMI(2) = std(NMIi);
    Purity(1) =mean(Purityi); Purity(2)=std(Purityi);
    
    F_array_avg =[F_array_avg;F(1)];
    P_array_avg =[P_array_avg;P(1)];
    R_array_avg = [R_array_avg;R(1)];
    nmi_array_avg =[nmi_array_avg;nmi(1)];
    avgent_array_avg=[avgent_array_avg;avgent(1)];
    AR_array_avg =[AR_array_avg;AR(1)];
    ACC_array_avg=[ACC_array_avg;ACC(1)];
    Purity_array_avg=[Purity_array_avg;Purity(1)];
    
    F_array_std =[F_array_std;F(2)];
    P_array_std =[P_array_std;P(2)];
    R_array_std = [R_array_std;R(2)];
    nmi_array_std =[nmi_array_std;nmi(2)];
    avgent_array_std=[avgent_array_std;avgent(2)];
    AR_array_std =[AR_array_std;AR(2)];
    ACC_array_std=[ACC_array_std;ACC(2)];
    Purity_array_std=[Purity_array_std;Purity(2)];
    pr_array =[pr_array;step];
    step =step+0.1;
end

fprintf('======================================\n');
fprintf('running MMSC for SensIT Vehicle\n');
fprintf('neurons =%d\n',nurons);
%% Printf the best result
step_ACC = pr_array(find(ACC_array_avg ==max(ACC_array_avg)));
step_NMI = pr_array(find(nmi_array_avg ==max(nmi_array_avg)));
step_Purity = pr_array(find(Purity_array_avg ==max(Purity_array_avg)));
fprintf('step_ACC=%f, step_NMI=%f, step_Purity=%f\n',step_ACC(1),step_NMI(1),step_Purity(1));
max_acc = max(ACC_array_avg);
std_acc = ACC_array_std(find(ACC_array_avg==max(ACC_array_avg)));
max_nmi = max(nmi_array_avg);
std_nmi =nmi_array_std(find(nmi_array_avg==max(nmi_array_avg)));
max_purity =max(Purity_array_avg);
std_purity =Purity_array_std(find(Purity_array_avg==max(Purity_array_avg)));
fprintf('ACC=%0.4f(%0.4f),nmi score=%0.4f(%0.4f),Purity=%0.4f(%0.4f)\n',max_acc(1),std_acc(1),max_nmi(1),std_nmi(1),max_purity(1),std_purity(1));

save('./result/MMSC_SensITVehicle.mat','pr','neurons_i','ACC_array_avg','ACC_array_std','nmi_array_avg','nmi_array_std','Purity_array_avg','Purity_array_std');
toc;
