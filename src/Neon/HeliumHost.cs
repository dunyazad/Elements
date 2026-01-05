using System;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Input;
using System.Windows.Interop;

namespace Neon.Controls
{
    public sealed class HeliumHost : HwndHost
    {
        private IntPtr _hwnd = IntPtr.Zero;

        // 윈도우 메시지 상수 정의
        private const int WM_KEYDOWN = 0x0100;
        private const int WM_KEYUP = 0x0101;
        private const int WM_SYSKEYDOWN = 0x0104;
        private const int WM_SYSKEYUP = 0x0105;

        private const int WM_MOUSEMOVE = 0x0200;
        private const int WM_LBUTTONDOWN = 0x0201;
        private const int WM_LBUTTONUP = 0x0202;
        private const int WM_RBUTTONDOWN = 0x0204;
        private const int WM_RBUTTONUP = 0x0205;
        private const int WM_MBUTTONDOWN = 0x0207;
        private const int WM_MBUTTONUP = 0x0208;
        private const int WM_MOUSEWHEEL = 0x020A; // Vertical
        private const int WM_MOUSEHWHEEL = 0x020E; // Horizontal

        public HeliumHost()
        {
            this.Focusable = true;
        }

        // 마우스 클릭 시 포커스를 가져와야 키보드 입력을 받을 수 있음
        protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonDown(e);
            this.Focus();
        }

        protected override void OnMouseRightButtonDown(MouseButtonEventArgs e)
        {
            base.OnMouseRightButtonDown(e);
            this.Focus();
        }

        protected override HandleRef BuildWindowCore(HandleRef hwndParent)
        {
            _hwnd = NativeMethods.CreateWindowEx(
            0,
            "STATIC",
            "",
            NativeMethods.WS_CHILD |
            NativeMethods.WS_VISIBLE |
            NativeMethods.WS_CLIPCHILDREN |
            NativeMethods.WS_CLIPSIBLINGS,
            0,
            0,
            (int)ActualWidth,
            (int)ActualHeight,
            hwndParent.Handle,
            IntPtr.Zero,
            IntPtr.Zero,
            IntPtr.Zero);

            if (_hwnd == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create Helium host window.");

            HeliumNative.He_Initialize(_hwnd, 0);

            return new HandleRef(this, _hwnd);
        }

        protected override void DestroyWindowCore(HandleRef hwnd)
        {
            HeliumNative.He_Shutdown();
            NativeMethods.DestroyWindow(hwnd.Handle);
            _hwnd = IntPtr.Zero;
        }

        protected override void OnRenderSizeChanged(SizeChangedInfo sizeInfo)
        {
            base.OnRenderSizeChanged(sizeInfo);

            if (_hwnd != IntPtr.Zero)
            {
                // DPI Scale 계산
                double dpiScaleX = 1.0;
                double dpiScaleY = 1.0;

                PresentationSource source = PresentationSource.FromVisual(this);
                if (source != null && source.CompositionTarget != null)
                {
                    dpiScaleX = source.CompositionTarget.TransformToDevice.M11;
                    dpiScaleY = source.CompositionTarget.TransformToDevice.M22;
                }

                int width = (int)(sizeInfo.NewSize.Width * dpiScaleX);
                int height = (int)(sizeInfo.NewSize.Height * dpiScaleY);

                HeliumNative.He_Resize(width, height);
            }
        }

        protected override IntPtr WndProc(IntPtr hwnd, int msg, IntPtr wParam, IntPtr lParam, ref bool handled)
        {
            return base.WndProc(hwnd, msg, wParam, lParam, ref handled);
        }
    }
}
