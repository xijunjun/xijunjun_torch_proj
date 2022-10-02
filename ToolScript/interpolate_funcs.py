
https://vimsky.com/examples/usage/python-scipy.interpolate.CubicSpline.html

# x = global_etou_pts_np[:,0]
# y = global_etou_pts_np[:, 1]
# x=np.append(x,x[0])
# y=np.append(y, y[0])
#
#
# cs = interpolate.CubicSpline(x, y, bc_type='periodic')
# # cs = interpolate.CubicSpline(x, y)
# xx = np.linspace(global_etou_pts[0][0], global_etou_pts[-1][0], 100)
# yy=cs(xx)
# fitpts=[xx,yy]
# fitpts_np=np.array(fitpts).T
# np.swapaxes(fitpts_np,0,1).reshape(-1,2)
# draw_pts(halfheadalign_visual, list(fitpts_np), 4, (0, 0, 255), 4)
# line_ptlist(halfheadalign_visual, list(fitpts_np), (0, 0, 255),4)


inds = np.array([i for i in range(0, num_global_etou_pts + 1)])
# inds=np.append(inds,inds[0])
x = global_etou_pts_np[:, 0]
y = global_etou_pts_np[:, 1]
x = np.append(x, x[0])
y = np.append(y, y[0])

csx = interpolate.CubicSpline(inds, x, bc_type='periodic')
csy = interpolate.CubicSpline(inds, y, bc_type='periodic')
indfit = np.linspace(0, num_global_etou_pts, 100)

xfit = csx(indfit)
yfit = csy(indfit)
fitpts = [xfit, yfit]
fitpts_np = np.array(fitpts).T
np.swapaxes(fitpts_np, 0, 1).reshape(-1, 2)
draw_pts(halfheadalign_visual, list(fitpts_np), 4, (0, 0, 255), 4)
line_ptlist(halfheadalign_visual, list(fitpts_np), (0, 0, 255), 4)

x = global_etou_pts_np[:, 0]
y = global_etou_pts_np[:, 1]
tck = interpolate.splrep(x, y)
xx = np.linspace(global_etou_pts[0][0], global_etou_pts[-1][0], 100)
yy = interpolate.splev(xx, tck, der=0)

fitpts = [xx, yy]
fitpts_np = np.array(fitpts).T
np.swapaxes(fitpts_np, 0, 1).reshape(-1, 2)
draw_pts(halfheadalign_visual, list(fitpts_np), 2, (0, 0, 0), 2)
line_ptlist(halfheadalign_visual, list(fitpts_np), (0, 0, 0), 2)