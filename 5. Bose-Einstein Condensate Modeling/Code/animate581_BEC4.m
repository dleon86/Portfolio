function [M,h,hf] = animate581_BEC4(psi,psi_f,X,Y,Z,Kx,Ky,Kz,n,tspan,L,Lk,title,A,B)

%     v = VideoWriter(title);
%     v.FrameRate = 30;
%     open(v);
    
    % determine the max and min value of omega, which change with each initial
    % condition, for plotting purposes 
    wmax = ceil(max(max(max(max(psi)))));
    wmin = floor(min(min(min(min(psi)))));
    
    % make vector to rotate coordinates
%     rot = linspace(30,50,length(tspan));
    rot = 45.*ones(length(tspan),1); % choose for no rotation of axes
    
    % initiallize figure
    h = figure;
    % customize the figure axes
    axis tight manual
    ax = gca;
    ax.NextPlot = 'replaceChildren';

    % preallocate array to store movie frames
    M(length(tspan)) = struct('cdata',[],'colormap',[]);

    % prepare figure properties, including setting the background to white
    set(gcf,'units','normalized','outerposition',[0.2 -0.7 0.6 1],'color','w')
    h.Visible = 'off';
    
    
    
%     % set the colormap
%     colormap(turbo);%lines);%jet)%hot)
    subplot(2,2,1)
    p1 = patch(isosurface(X,Y,Z,real(psi(:,:,:,1))),'facecolor','g','edgecolor','none'); %

    colormap(turbo)
    daspect([1 1 1]); 
    view(3);
    axis tight;
    camlight;
    lighting gouraud;
    camproj perspective;
    set(gca,'xlim',[-L/2 L/2],'ylim',[-L/2 L/2],'zlim',[-L/2 L/2])
    txt2a = sprintf('$Re(\\psi)$',n);
    text(-L/2,L/2,L/2,txt2a,'fontsize',20,'fontweight','bold','interpreter','latex','HorizontalAlignment','center');
    
%     title({['$Re(\psi)$ in Spatial Domain' ];[txt1a]},'fontsize',20,'fontweight','bold','interpreter','latex')

    
    subplot(2,2,2)
    p2 = patch(isosurface(Kx,Ky,Kz,real(psi_f(:,:,:,1))),'facecolor','c','edgecolor','none'); %
%     txt3a = sprintf('$Re(\\hat{\\psi})$',n);
%     text(-Lk,Lk,Lk,txt3a,'fontsize',20,'fontweight','bold','interpreter','latex');
    
    colormap(turbo)
    daspect([1 1 1]); 
    view(3);
    axis tight;
    camlight;
    lighting gouraud;
    camproj perspective;
    set(gca,'xlim',[-Lk Lk],'ylim',[-Lk Lk],'zlim',[-Lk Lk]) 
    txt3a = sprintf('$Re(\\hat{\\psi})$',n);
    text(-Lk,Lk,Lk,txt3a,'fontsize',20,'fontweight','bold','interpreter','latex','HorizontalAlignment','center');
    
    subplot(2,2,3)
    p3 = patch(isosurface(X,Y,Z,imag(psi(:,:,:,1))),'facecolor','r','edgecolor','none'); %
%     txt4a = sprintf('$Im(\\psi)$',n);
%     text(-L/2,L/2,L/2,txt4a,'fontsize',20,'fontweight','bold','interpreter','latex');

    colormap(turbo)
    daspect([1 1 1]); 
    view(3);
    axis tight;
    camlight;
    lighting gouraud;
    camproj perspective;
    set(gca,'xlim',[-L/2 L/2],'ylim',[-L/2 L/2],'zlim',[-L/2 L/2])
    txt4a = sprintf('$Im(\\psi)$',n);
    text(-L/2,L/2,L/2,txt4a,'fontsize',20,'fontweight','bold','interpreter','latex','HorizontalAlignment','center');
    
    subplot(2,2,4)
    p4 = patch(isosurface(Kx,Ky,Kz,imag(psi_f(:,:,:,1))),'facecolor','m','edgecolor','none'); %
%     txt5a = sprintf('$Im(\\hat{\\psi})$',n);
%     text(-Lk,Lk,Lk,txt5a,'fontsize',20,'fontweight','bold','interpreter','latex');
    
    colormap(turbo)
    daspect([1 1 1]); 
    view(3);
    axis tight;
    camlight;
    lighting gouraud;
    camproj perspective;    
%     set(gca,'xlim',[-Lk/2 Lk/2],'ylim',[-Lk/2 Lk/2],'zlim',[-Lk/2 Lk/2])
    set(gca,'xlim',[-Lk Lk],'ylim',[-Lk Lk],'zlim',[-Lk Lk])
    txt5a = sprintf('$Im(\\hat{\\psi})$',n);
    text(-Lk,Lk,Lk,txt5a,'fontsize',20,'fontweight','bold','interpreter','latex','HorizontalAlignment','center');
%     txt6a = sprintf('$Im(\\hat{\\psi})$',n);
%     text(-Lk,Lk,Lk,txt6a,'fontsize',20,'fontweight','bold','interpreter','latex');
    
    r = 1;
    for j = 1:length(tspan)
%         clf
        %                                    display the current time
%         hold off
%         txt1 = sprintf('Fourier modes: %d \n $t = \\hspace{3pt}$ %d',n,tspan(j));
        txt1 = sprintf('Fourier modes: %d \n $(t_i,t_f,\\Delta t)=($%d,%d,%G$)$ \n $A = ($%G,%G,%G$)$ \n $B = ($%G,%G,%G$)$',...
            n,0,tspan(end),mean(diff(tspan)),A(1),A(2),A(3),B(1),B(2),B(3));
        text(-0.35,1.2,txt1,'fontsize',20,'interpreter','latex','units','normalized','HorizontalAlignment','center');
        
        txtx = sprintf('$$i\\psi_t = -\\frac{1}{2}\\nabla^2\\psi + |\\psi|^2\\psi - [A_1 \\sin^2(x) + B_1][A_2 \\sin^2(y) + B_2][A_3 \\sin^2(z) + B_3]\\psi$$');
        text(-0.35,2.6,txtx,'fontsize',20,'interpreter','latex','units','normalized','HorizontalAlignment','center');
%         text(0,r,r,sprintf('$t =\\hspace{0.3em} $%0.1f',tspan(j)),'interpreter','latex', 'fontsize',28,'units','normalized')
%         txt = sprintf('$t =\\hspace{0.3em} $%0.1f',tspan(j));
%         title(sprintf('$t =\\hspace{0.3em} $%0.1f',tspan(j)),'interpreter','latex','fontweight','bold', 'fontsize',28)
%         hold on
        view(rot(j),25);
        subplot(2,2,1)
        [faces_tr, vertices_tr] = isosurface(X,Y,Z,real(psi(:,:,:,j)));
        set(p1, 'Faces',faces_tr,'Vertices',vertices_tr);
        set(gca,'xlim',[-L/2 L/2],'ylim',[-L/2 L/2],'zlim',[-L/2 L/2])
        
        % set axis labels and fonts
        set(gca,'fontsize',16)
        xlabel('$x$','interpreter','latex','fontsize', 26); 
        ylabel('$y$','interpreter','latex','fontsize',26);
        zlabel('$z$ ','rotation',0,'interpreter','latex','fontsize',26);
        
        view(rot(j),25);
        subplot(2,2,2)
        [faces_fr, vertices_fr] = isosurface(Kx,Ky,Kz,real(psi_f(:,:,:,j)));

        set(p2, 'Faces',faces_fr,'Vertices',vertices_fr);
%         set(gca,'xlim',[-Lk/2 Lk/2],'ylim',[-Lk/2 Lk/2],'zlim',[-Lk/2 Lk/2])
        set(gca,'xlim',[-Lk Lk],'ylim',[-Lk Lk],'zlim',[-Lk Lk])

        % set axis labels and fonts
        set(gca,'fontsize',16)
        xlabel('$K_x$','interpreter','latex','fontsize', 26); 
        ylabel('$K_y$','interpreter','latex','fontsize',26);
        zlabel('$K_z$ ','rotation',0,'interpreter','latex','fontsize',26);
        
        view(rot(j),25);
        subplot(2,2,3)
        [faces_ti, vertices_ti] = isosurface(X,Y,Z,imag(psi(:,:,:,j)));
        set(p3, 'Faces',faces_ti,'Vertices',vertices_ti);
         set(gca,'xlim',[-L/2 L/2],'ylim',[-L/2 L/2],'zlim',[-L/2 L/2])

        
        % set axis labels and fonts
        set(gca,'fontsize',16)
        xlabel('$x$','interpreter','latex','fontsize', 26); 
        ylabel('$y$','interpreter','latex','fontsize',26);
        zlabel('$z$ ','rotation',0,'interpreter','latex','fontsize',26);
        
        view(rot(j),25);
        subplot(2,2,4)
        [faces_fi, vertices_fi] = isosurface(Kx,Ky,Kz,imag(psi_f(:,:,:,j)));

        set(p4, 'Faces',faces_fi,'Vertices',vertices_fi);
%         set(gca,'xlim',[-Lk/2 Lk/2],'ylim',[-Lk/2 Lk/2],'zlim',[-Lk/2 Lk/2])
        set(gca,'xlim',[-Lk Lk],'ylim',[-Lk Lk],'zlim',[-Lk Lk])
        
        % set axis labels and fonts
        set(gca,'fontsize',16)
        xlabel('$K_x$','interpreter','latex','fontsize', 26); 
        ylabel('$K_y$','interpreter','latex','fontsize',26);
        zlabel('$K_z$ ','rotation',0,'interpreter','latex','fontsize',26);        
    
        drawnow
        pause(0.1)
        M(j) = getframe(h);

        frame = getframe(gcf);
        img =  frame2im(frame);
        [img,cmap] = rgb2ind(img,64);

        % Set the filename 1-4 based on which conditions are being used.
        if j == 1
            imwrite(img,cmap, title,'gif','LoopCount',Inf,'DelayTime',0.3);
        else
            imwrite(img,cmap, title,'gif','WriteMode','append','DelayTime',0.25);
        end
        pause(0.25);
%



%         writeVideo(v,M(j));
        
    end

    % this is needed to properly display the plots within the figure
    [ht w d] = size(M(1).cdata);
    hf = figure;
    set(hf,'position',[1 1 w ht]);
    axis off

%     close(v);
    
    % play video
    movie(hf,M);
%     movie(M);
end