using System;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Interop;

namespace Neon.Controls
{
    public sealed class HeliumHost : HwndHost
    {
        private IntPtr _hwnd = IntPtr.Zero;

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

            HeliumNative.Helium_Initialize(_hwnd);

            return new HandleRef(this, _hwnd);
        }

        protected override void DestroyWindowCore(HandleRef hwnd)
        {
            HeliumNative.Helium_Shutdown();
            NativeMethods.DestroyWindow(hwnd.Handle);
            _hwnd = IntPtr.Zero;
        }

        protected override void OnRenderSizeChanged(SizeChangedInfo sizeInfo)
        {
            base.OnRenderSizeChanged(sizeInfo);

            if (_hwnd != IntPtr.Zero)
            {
                HeliumNative.Helium_Resize(
                    (int)sizeInfo.NewSize.Width,
                    (int)sizeInfo.NewSize.Height);
            }
        }
    }
}
