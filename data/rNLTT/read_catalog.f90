      PROGRAM read_catalog

      implicit none

      real :: x3,x4,x5,x6,x7,x8,x9,x10,x14,x15,x16,x17,x18,x19
      real :: x24,x25,x26,x30,x31,x32,x33,x34,x35,x38
      real :: x39,x40,x41
      integer :: x1,x11,x12,x13,x20,x22,x23,x27,x28,x29,x36,x37,x42
      integer :: io
      character (len=10) :: x2,x21,x43,x44


      open(22,file='catalog.dat',status='old',iostat=io)
      open(23,file='catalog_tabs.dat',action='write',status='replace')

100   format(i5,a1,2f11.6,2f8.4,2f7.4,2f6.2,1x,3i1,2f10.5,2f8.4,2f5.1,i7,1x,a12,2i10, &
        & 2f5.1,f9.3,i2,2i1,2f10.5,3f7.3,f9.3,i2,i6,f7.1,f6.1,f7.1,f6.1,i2,1x,a12,1x,a5)


      write(23,*) "NLTT	ra	dec	mu_ra	mu_dec	mu_ra_err	mu_dec_err", &
        & "	V	B	R	J	H	K"

      do
        read(22,100) x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19, &
          & x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,x34,x35,x36,x37,x38, &
          & x39,x40,x41,x42,x43,x44


        if (io<0) exit

        write(23,*) x1,x3,x4,x5,x6,x7,x8,x9,x24,x25,x32,x33,x34
      end do

!      write(*,*) x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19, &
!        & x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,x34,x35,x36,x37,x38, &
!        & x39,x40,x41,x42,x43,x44




      close(22)


      STOP
      END
