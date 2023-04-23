% ================ Limpa todas variáveis armazenadas ======================
clear all
clc
% ====================== Entrada de dados =================================
% --> Carrega base de dados - 16 Parâmetros
load('Data2.mat'); % Armazena base na variável Data

% --> Separa a quantidade de entradas de cada classe
nt = [70 70 70]; 

% --> Dimensão da matriz de entrada
[lin, col] = size(Data);

% ========================== KFOLD ========================================
Kfold = 10;
lenset = lin/Kfold;
ntTrain = nt-(lin/(3*Kfold));
ntTest = nt-ntTrain;

t1 = 1; t2 = nt(1)+1; t3 = nt(1)+nt(2)+1; tP = nt(1)/Kfold-1;
for i=1:Kfold
    d(1:(lenset),i) = [t1:(t1+tP),t2:(t2+tP),t3:(t3+tP)]';
    t1 = t1+tP+1;
    t2 = t2+tP+1;
    t3 = t3+tP+1;
end

% Gabarito dos subconjuntos
tc = nt(1)/Kfold;
for i=1:Kfold
    c(1:tc,i) = 1;
    c((tc+1):(tc*2),i) = 2;
    c((tc*2+1):(tc*3),i) = 3;    
end

%--> Separa dados de treinamento e de teste
for i = 1:Kfold 
    trainData = Data;
    for j = d(:,i)
        trainData(j,:) = [];
        testData = Data(j,:);
    end
    
trainData = zscore(trainData);
testData = zscore(testData);

% ========================== LDA ==========================================
% [L,K,V]=lda(X,ns,nt,n)
%
% L - Eigenvectors(sorted) of matrix Z. Each column represents an eigenvector.
% K - Eigenvalues (sorted) of matrix Z. It is a column vector.
% V - Variance explained by each corresponding eigenvalues.
%
% X - Matrix containing the train. set, whose each line points to a sample data.
% ns- Number of distinct subjects (groups) readen.
% nt- Vector of the number of photos of each subject used as training sample.
% n - Reduction of dimension (n <= ns-1).
% 
% Carlos Thomaz, DOC-IC/LONDON, 15/01/2004.[L,K,V] = mlda(trainNorm,3,nt,2); 
[L,K,V]=lda(trainData,3,ntTrain,2); % Calcula a LDA
Z = trainData * L; % Projeta os dados nos Autovetores L

Ztest = testData * L; % Projeta os dados de teste nos Autovetores L (de treinamento)

% ======================== Distância euclidiana ===========================
data1 = Z(1:ntTrain(1),:);                                                  % Amostras do grupo 1 - Neutro
data2 = Z((ntTrain(2)+1):(ntTrain(1)+ntTrain(2)),:);                        % Amostras do grupo 2 - Washout
data3 = Z((ntTrain(1)+ntTrain(2)+1):(ntTrain(1)+ntTrain(2)+ntTrain(3)),:);  % Amostras do grupo 3 - Blurry

mG = [mean(data1); mean(data2); mean(data3)]; % Media dos grupos

data4 = Ztest(1:ntTest(1),:);                                               % Amostras do grupo 1 - Neutro
data5 = Ztest((ntTest(2)+1):(ntTest(1)+ntTest(2)),:);                       % Amostras do grupo 2 - Washout
data6 = Ztest((ntTest(1)+ntTest(2)+1):(ntTest(1)+ntTest(2)+ntTest(3)),:);   % Amostras do grupo 3 - Blurry

N = []; W = []; B = [];
len = size(Ztest);
for v = 1:len(1)    % Cálculo da distâcia da média 
    N(v,1) = sqrt(((mG(1,1)-Ztest(v,1))^2)+((mG(1,2)-Ztest(v,2))^2));      
    W(v,1) = sqrt(((mG(2,1)-Ztest(v,1))^2)+((mG(2,2)-Ztest(v,2))^2));
    B(v,1) = sqrt(((mG(3,1)-Ztest(v,1))^2)+((mG(3,2)-Ztest(v,2))^2));
end

CLASS = [];                                 % Classificador pela distância
for w = 1:len(1)
    if N(w,1) <= W(w,1) && N(w,1) <= B(w,1)     % Condição para classe 1
        CLASS(w,1) = 1;                         % Classifica como classe 1
    elseif W(w,1) < N(w,1) && W(w,1) < B(w,1)   % Condição para classe 2
        CLASS(w,1) = 2;                         % Classifica como classe 2
    else
        CLASS(w,1) = 3;                         % Classifica como classe 3
    end
end

FIM = [CLASS c(:,i)];                    % Matriz de verificação

ACERTO = 0;
for z=1:len(1)                          % Conferência de acertos
    if FIM(z,1)==FIM(z,2)
        ACERTO = ACERTO+1;
    end
end
TAXA(i) = ACERTO/len(1);                   % Taxa de acerto global

% ============================== Gráfico ==================================
figure
XMI = -0.5; XMA = 0.5 ;
YMI = -2.3; YMA = 1.5;

hAxes = axes('NextPlot','add',...           % Add subsequent plots to the axes,
             'DataAspectRatio',[1 1 1],...  % match the scaling of each axis,             
             'XLim',[(XMI-1) (XMA+1)],...   % set the x axis limit,
             'YLim',[(YMI-1) (YMA+1)],...   % set the y axis limit (tiny!),
             'Color','none');               % and don't use a background color
%            'Position', [0.1 0.1 0.8 0]); % Set the position onf the plot.

p1 = plot(data1(:,1),data1(:,2),'ro','MarkerSize',10); hold on   % Plot data set 1
p2 = plot(data2(:,1),data2(:,2),'bo','MarkerSize',10); hold on   % Plot data set 2
p3 = plot(data3(:,1),data3(:,2),'go','MarkerSize',10); hold on   % Plot data set 3
p4 = plot(data4(:,1),data4(:,2),'^','MarkerEdge',[0 0 0],'MarkerFace',[0.8 0.2 0.2],'MarkerSize',7); hold on   % Plot data set 1
p5 = plot(data5(:,1),data5(:,2),'^','MarkerEdge',[0 0 0],'MarkerFace',[0.2 0.2 0.8],'MarkerSize',7); hold on   % Plot data set 2
p6 = plot(data6(:,1),data6(:,2),'^','MarkerEdge',[0 0 0],'MarkerFace',[0.2 0.8 0.2],'MarkerSize',7); hold on   % Plot data set 3
p7 = plot(mG(1,1),mG(1,2),'s','MarkerEdge',[0 0 0],'MarkerFace',[0.8 0.2 0.2],'MarkerSize',10); hold on % Plot Media dos grupos
p8 = plot(mG(2,1),mG(2,2),'s','MarkerEdge',[0 0 0],'MarkerFace',[0.2 0.2 0.8],'MarkerSize',10); hold on % Plot Media dos grupos
p9 = plot(mG(3,1),mG(3,2),'s','MarkerEdge',[0 0 0],'MarkerFace',[0.2 0.8 0.2],'MarkerSize',10); hold on % Plot Media dos grupos
p10 = plot(mG(1,1),mG(1,2),'s','MarkerEdge',[0 0 0],'MarkerSize',10); hold on % Plot Media dos grupos  
p11 = plot(data4(:,1),data4(:,2),'^','MarkerEdge',[0 0 0],'MarkerSize',7); hold on   % Plot data set 1
leg = legend([p1(1), p2(1), p3(1), p10(1), p11(1)],...
'Neutral', 'Washout', 'Blurry','Centroíde','Predição','Acurácia');
set(gca,'color','w'); % Background do gráfico branco
set(leg,'color','w'); % Background da legenda branco
set(gcf,'color','w'); % Background do plot branco
xlabel('Hiperplano discriminante 1');
ylabel('Hiperplano discriminante 2');
end

acuracia = mean(TAXA)
