classdef Data
    properties
        X_max;
        X_min;
        Y_max;
        Y_min;
    end

    methods
        function outputArg = Data()
 
        end

        function [X,y,weight]=data_init(~,X,y)
            X=[ones(size(X,1),1),X];
            %weight=normrnd(0,0.1,[size(X,2),1]);
            weight=zeros([size(X,2),1]);
        end

        function [X,Y,this]=normalization(this,X,Y)
            this.X_max=max(X);
            this.X_min=min(X);
            this.Y_max=max(Y);
            this.Y_min=min(Y);
            for i=1:size(X,2)
                X(:,i)=(X(:,i)-min(X(:,i)))./(max(X(:,i))-min(X(:,i)));
            end
            for i=1:size(Y,2)
                Y(:,i)=(Y(:,i)-min(Y(:,i)))./(max(Y(:,i))-min(Y(:,i)));
            end
        end

        function [X,Y,predictions]=inverse_normalization(this,X,Y,predictions)
            for i=1:size(X,2)
                X(:,i)=X(:,i).*(this.X_max(i)-this.X_min(i))+this.X_min(i);
            end
            for i=1:size(Y,2)
                Y(:,i)=Y(:,i).*(this.Y_max(i)-this.Y_min(i))+this.Y_min(i);
            end
            if(nargin==4)
                for i=1:size(predictions,2)
                    predictions(:,i)=predictions(:,i).*(this.Y_max(i)-this.Y_min(i))+this.Y_min(i);
                end
            end
        end

        function [X,y]=data_iter(~,batch_size,features,labels)
            %生成批量数据
            num_examples=size(features,1);
            indices=randperm(num_examples);
            k=1;
            for i=1:batch_size:num_examples
                batch_indices=indices(i:min([i+batch_size-1,num_examples]));
                X(:,:,k)=features(batch_indices,:);
                y(:,k)=labels(batch_indices);
                k=k+1;
            end

        end

        function [X,y] = synthetic_linear_data(~,w,b,num_examples)
            %生成虚拟线性数据
            X=normrnd(0,1,[num_examples size(w,1)]);
            y=X*w+b;
            y=y+normrnd(0,0.01,size(y));
        end

        function [X,f] = synthetic_nolinear_data(~,num_examples)
            X=unifrnd(-2*pi,2*pi,[num_examples 2]);
            f=2*X(:,1).^2+sin(X(:,2)+pi/4);
        end

        function data_scatter(~,x,y)
            plot(x,y,'.','MarkerSize',8);
        end       
    end
end