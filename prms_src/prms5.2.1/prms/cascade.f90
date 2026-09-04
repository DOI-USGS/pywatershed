!***********************************************************************
! Determines computational order of the HRUs and ground-water
! reservoirs for routing flow downslope
!***********************************************************************
      MODULE PRMS_CASCADE
      IMPLICIT NONE
!   Local Variables
      character(len=*), parameter :: MODDESC = 'Cascading Flow'
      character(len=*), parameter :: MODNAME = 'cascade'
      character(len=*), parameter :: Version_cascade = '2021-08-13'
      INTEGER, SAVE :: MSGUNT
      INTEGER, SAVE :: Iorder, Igworder, Ndown
!   Computed Variables
      INTEGER, SAVE, ALLOCATABLE :: Hru_down(:, :), Gwr_down(:, :)
      INTEGER, SAVE, ALLOCATABLE :: Ncascade_hru(:), Ncascade_gwr(:)
      REAL, SAVE, ALLOCATABLE :: Cascade_area(:, :), Hru_down_fracwt(:, :)
      REAL, SAVE, ALLOCATABLE :: Cascade_gwr_area(:, :), Gwr_down_frac(:, :)
      REAL, SAVE, ALLOCATABLE :: Hru_down_frac(:, :)
!     REAL, SAVE, ALLOCATABLE :: Gwr_down_fracwt(:, :)
! hru_down_frac: Fraction of HRU area used to compute flow routed
!                to a downslope HRU or stream segment.
! hru_down_fracwt: HRU area fraction, area weighted by downslope HRU
!                  area, used to compute flow routed to a downslope
!                  HRU or stream segment.
! gwr_down_fracwt: GWR area fraction, area weighted by downslope GWR
!                  area, used to compute flow routed to a downslope
!                  GWR or stream segment.
! gwr_down_frac: Fraction of GWR area used to compute flow routed
!                to a downslope cascade area or stream segment from
!                each cascade area of an GWR.
! cascade_area: Cascade area within an HRU.
! cascade_gwr_area: Cascade area within an GWR.
! hru_down: Indices of the downslope HRUs or stream segments to
!           which the cascade area routes flow.
! gwr_down: Indices of the downslope GWRs to which the cascade
!           area routes flow.
!   Declared Parameters
      INTEGER, SAVE :: Cascade_flg, Circle_switch
      REAL, SAVE :: Cascade_tol
      INTEGER, SAVE, ALLOCATABLE :: Hru_up_id(:), Hru_strmseg_down_id(:), Hru_down_id(:)
      INTEGER, SAVE, ALLOCATABLE :: Gw_up_id(:), Gw_strmseg_down_id(:), Gw_down_id(:)
      INTEGER, SAVE, ALLOCATABLE :: Hru_segment(:)
      REAL, SAVE, ALLOCATABLE:: Hru_pct_up(:), Gw_pct_up(:)
      END MODULE PRMS_CASCADE

!***********************************************************************
!     Main cascade routine
!***********************************************************************
      INTEGER FUNCTION cascade()
      USE PRMS_CONSTANTS, ONLY: DECL, INIT, CLEAN
      USE PRMS_MODULE, ONLY: Process_flag
      IMPLICIT NONE
! Functions
      INTEGER, EXTERNAL :: cascdecl, cascinit, cascclean
!***********************************************************************
      cascade = 0

      IF ( Process_flag==DECL ) THEN
        cascade = cascdecl()
      ELSEIF ( Process_flag==INIT ) THEN
        cascade = cascinit()
      ELSEIF ( Process_flag==CLEAN ) THEN
        cascade = cascclean()
      ENDIF

      END FUNCTION cascade

!***********************************************************************
!     cascdecl - set up parameters for cascading flow
!   Declared Parameters
!     hru_up_id, hru_down_id, hru_pct_up, hru_strmseg_down_id
!     gw_up_id, gw_down_id, gw_pct_up, gw_strmseg_down_id
!     hru_area, cascade_tol, cascade_flg, circle_switch
!***********************************************************************
      INTEGER FUNCTION cascdecl()
      USE PRMS_CONSTANTS, ONLY: DOCUMENTATION, CASCADE_OFF, CASCADE_HRU_SEGMENT, CASCADE_NORMAL, CASCADEGW_OFF
      USE PRMS_MODULE, ONLY: Nhru, Ngw, Ncascade, Ncascdgw, Model, Print_debug, Cascade_flag, Cascadegw_flag
      USE PRMS_CASCADE
      IMPLICIT NONE
! Functions
      INTRINSIC :: INDEX
      INTEGER, EXTERNAL :: declparam
      EXTERNAL :: read_error, print_module, PRMS_open_module_file
!***********************************************************************
      cascdecl = 0

      CALL print_module(MODDESC, MODNAME, Version_cascade)

      IF ( Cascade_flag>CASCADE_OFF .OR. Model==DOCUMENTATION ) ALLOCATE ( Ncascade_hru(Nhru) )

      IF ( Cascadegw_flag>CASCADEGW_OFF .OR. Model==DOCUMENTATION ) ALLOCATE ( Ncascade_gwr(Ngw) )

      IF ( Print_debug==13 ) CALL PRMS_open_module_file(MSGUNT, 'cascade.msgs')

! declare HRU cascade parameters
      IF ( Cascade_flag>CASCADE_OFF ) THEN
        ALLOCATE ( Hru_up_id(Ncascade) )
        ALLOCATE ( Hru_strmseg_down_id(Ncascade) )
        ALLOCATE ( Hru_down_id(Ncascade) )
        ALLOCATE ( Hru_pct_up(Ncascade) )
      ENDIF
      IF ( Cascade_flag==CASCADE_NORMAL .OR. Model==DOCUMENTATION ) THEN
        IF ( declparam(MODNAME, 'hru_up_id', 'ncascade', 'integer', &
     &       '0', 'bounded', 'nhru', &
     &       'Index of HRU containing cascade area', &
     &       'Index of HRU containing cascade area', &
     &       'none')/=0 ) CALL read_error(1, 'hru_up_id')

        IF ( declparam(MODNAME, 'hru_strmseg_down_id', 'ncascade', 'integer', &
     &       '0', 'bounded', 'nsegment', &
     &       'Stream segment index that cascade area contributes flow', &
     &       'Index number of the stream segment that cascade area contributes flow', &
     &       'none')/=0 ) CALL read_error(1, 'hru_strmseg_down_id')

        IF ( declparam(MODNAME, 'hru_down_id', 'ncascade', 'integer', &
     &       '0', 'bounded', 'nhru', &
     &       'HRU index of downslope HRU', &
     &       'Index number of the downslope HRU to which the upslope HRU contributes flow', &
     &       'none')/=0 ) CALL read_error(1, 'hru_down_id')

        IF ( declparam(MODNAME, 'hru_pct_up', 'ncascade', 'real', &
     &       '1.0', '0.0', '1.0', &
     &       'Fraction of HRU area associated with cascade area', &
     &       'Fraction of HRU area used to compute flow contributed'// &
     &       ' to a downslope HRU or stream segment for cascade area', &
     &       'decimal fraction')/=0 ) CALL read_error(1, 'hru_pct_up')
      ENDIF
      IF ( Cascade_flag==CASCADE_HRU_SEGMENT .OR. Model==DOCUMENTATION ) THEN ! use hru_segment to define simple cascades
        ALLOCATE ( Hru_segment(Nhru) )
        IF ( declparam(MODNAME, 'hru_segment', 'nhru', 'integer', &
     &       '0', 'bounded', 'nsegment', &
     &       'Segment index for HRU lateral inflows', &
     &       'Segment index to which an HRU contributes lateral flows'// &
     &       ' (surface runoff, interflow, and groundwater discharge)', &
     &       'none')/=0 ) CALL read_error(1, 'hru_segment')
      ENDIF
      IF ( Cascade_flag/=CASCADE_HRU_SEGMENT .OR. Model==DOCUMENTATION ) THEN
        IF ( declparam(MODNAME, 'cascade_tol', 'one', 'real', &
     &       '5.0', '0.0', '99.0', &
     &       'Cascade area below which a cascade link is ignored', &
     &       'Cascade area below which a cascade link is ignored', &
     &       'acres')/=0 ) CALL read_error(1, 'cascade_tol')

        IF ( declparam(MODNAME, 'cascade_flg', 'one', 'integer', &
     &       '0', '0', '1', &
     &       'Flag to indicate cascade type (0=allow many to many; 1=force one to one)', &
     &       'Flag to indicate cascade type (0=allow many to many; 1=force one to one)', &
     &       'none')/=0 ) CALL read_error(1, 'cascade_flg')

        IF ( declparam(MODNAME, 'circle_switch', 'one', 'integer', &
     &       '1', '0', '1', &
     &       'Switch to check for circles', &
     &       'Switch to check for circles (0=no check; 1=check)', &
     &       'none')/=0 ) CALL read_error(1, 'circle_switch')
      ENDIF

      IF ( Cascadegw_flag>CASCADEGW_OFF ) THEN
        ALLOCATE ( Gw_up_id(Ncascdgw) )
        ALLOCATE ( Gw_strmseg_down_id(Ncascdgw) )
        ALLOCATE ( Gw_down_id(Ncascdgw) )
        ALLOCATE ( Gw_pct_up(Ncascdgw) )
      ENDIF
      IF ( Cascadegw_flag==CASCADE_NORMAL .OR. Model==DOCUMENTATION ) THEN
! declare GWR cascade parameters
        IF ( declparam(MODNAME, 'gw_up_id', 'ncascdgw', 'integer', &
     &       '0', 'bounded', 'ngw', &
     &       'Index of GWR containing cascade area', &
     &       'Index of GWR containing cascade area', &
     &       'none')/=0 ) CALL read_error(1, 'gw_up_id')

        IF ( declparam(MODNAME, 'gw_strmseg_down_id', 'ncascdgw', 'integer', &
     &       '0', 'bounded', 'nsegment', &
     &       'Stream segment index that cascade area contributes flow', &
     &       'Index number of the stream segment that cascade area contributes flow', &
     &       'none')/=0 ) CALL read_error(1, 'gw_strmseg_down_id')

        IF ( declparam(MODNAME, 'gw_down_id', 'ncascdgw', 'integer', &
     &       '0', 'bounded', 'ngw', &
     &       'GWR index of downslope GWR', &
     &       'Index number of the downslope GWR to which the upslope GWR contributes flow', &
     &       'none')/=0 ) CALL read_error(1, 'gw_down_id')

        IF ( declparam(MODNAME, 'gw_pct_up', 'ncascdgw', 'real', &
     &       '1.0', '0.0', '1.0', &
     &       'Fraction of GWR area associated with cascade area', &
     &       'Fraction of GWR area used to compute flow contributed'// &
     &       ' to a downslope GWR or stream segment for cascade area', &
     &       'decimal fraction')/=0 ) CALL read_error(1, 'gw_pct_up')
      ENDIF

      END FUNCTION cascdecl

!***********************************************************************
!     cascinit - Initialize cascade module - get parameter values,
!***********************************************************************
      INTEGER FUNCTION cascinit()
      USE PRMS_CONSTANTS, ONLY: OFF, ERROR_cascades, CASCADE_OFF, CASCADE_HRU_SEGMENT, CASCADE_NORMAL, CASCADEGW_OFF
      USE PRMS_MODULE, ONLY: Ngw, Print_debug, Cascade_flag, Cascadegw_flag, Gwr_swale_flag
      USE PRMS_CASCADE
      USE PRMS_BASIN, ONLY: Active_hrus, Hru_route_order, Gwr_route_order, Active_gwrs, Gwr_type, Hru_type
      IMPLICIT NONE
! Functions
      INTEGER, EXTERNAL :: getparam
      EXTERNAL :: read_error, init_cascade, initgw_cascade
! Local Variables
      INTEGER :: i, j, k, ii, iret, itest
!***********************************************************************
      cascinit = 0

      IF ( Cascade_flag==CASCADE_HRU_SEGMENT ) THEN
        Cascade_tol = 5.0
        Cascade_flg = 1
        Circle_switch = 0
      ELSE
        IF ( getparam(MODNAME, 'cascade_tol', 1, 'real', Cascade_tol)/=0 ) CALL read_error(2, 'cascade_tol')
        IF ( getparam(MODNAME, 'cascade_flg', 1, 'integer', Cascade_flg)/=0 ) CALL read_error(2, 'cascade_flg')
        IF ( getparam(MODNAME, 'circle_switch', 1, 'integer', Circle_switch)/=0 ) CALL read_error(2, 'circle_switch')
      ENDIF

      IF ( Cascade_flag>CASCADE_OFF ) CALL init_cascade(itest)

      iret = 0
      IF ( Cascadegw_flag>CASCADEGW_OFF ) THEN
        ALLOCATE ( Gwr_down(Ndown,Ngw), Gwr_down_frac(Ndown,Ngw), Cascade_gwr_area(Ndown,Ngw) )
!        ALLOCATE ( Gwr_down_fracwt(Ndown,Ngw) )
        IF ( Cascadegw_flag==CASCADE_NORMAL ) THEN
          CALL initgw_cascade(iret)
          IF ( iret==1 ) ERROR STOP ERROR_cascades
        ELSE ! cascadegw_flag=2 (CASCADEGW_SAME) so GWR cascades set to HRU cascades
          Gwr_type = Hru_type
          Active_gwrs = Active_hrus
          Ncascade_gwr = Ncascade_hru
          Gwr_route_order = Hru_route_order
          Gwr_down = Hru_down
          DO i = 1, Ngw
            DO j = 1, Ndown
              Gwr_down_frac(j, i) = Hru_down_frac(j,i)
              Cascade_gwr_area(j, i) = Cascade_area(j,i)
            ENDDO
          ENDDO
        ENDIF
        IF ( Gwr_swale_flag==OFF ) THEN
          DO ii = 1, Active_gwrs
            i = Gwr_route_order(ii)
            IF ( Gwr_type(i)==3 ) THEN
              PRINT *, 'ERROR, GWR is a swale when gwr_swale_flag = 0, GWR:', i
              iret = 1
            ENDIF
          ENDDO
        ENDIF
      ENDIF
      IF ( itest/=0 .OR. iret/=0 ) ERROR STOP ERROR_cascades

      IF ( Print_debug==13 ) THEN
        IF ( Cascade_flag>CASCADE_OFF ) THEN
          WRITE ( MSGUNT, 9001 )
          k = 0
          DO ii = 1, Active_hrus
            i = Hru_route_order(ii)
            DO j = 1, Ncascade_hru(i)
              k = k + 1
              WRITE ( MSGUNT, * ) k, i, Hru_down(j, i), Hru_down_frac(j, i)*100.0
            ENDDO
          ENDDO
        ENDIF
        IF ( Cascadegw_flag>CASCADEGW_OFF ) THEN
          WRITE ( MSGUNT, 9002 ) 
          k = 0
          DO ii = 1, Active_gwrs
            i = Gwr_route_order(ii)
            DO j = 1, Ncascade_gwr(i)
              k = k + 1
              WRITE ( MSGUNT, * ) k, i, Gwr_down(j, i), Gwr_down_frac(j, i)*100.0
            ENDDO
          ENDDO
        ENDIF
        CLOSE ( MSGUNT )
      ENDIF

 9001 FORMAT (//, 18X, 'UP HRU', 4X, 'DOWN HRU    FRACTION')
 9002 FORMAT (//, 18X, 'UP GWR', 4X, 'DOWN GWR    FRACTION')

      END FUNCTION cascinit

!***********************************************************************
!     cascclean - deallocation arrays
!***********************************************************************
      INTEGER FUNCTION cascclean()
      USE PRMS_MODULE, ONLY: CASCADE_OFF, CASCADEGW_OFF
      USE PRMS_CASCADE
      USE PRMS_MODULE, ONLY: Cascade_flag, Cascadegw_flag
      IMPLICIT NONE
!***********************************************************************
      IF ( Cascade_flag>CASCADE_OFF ) THEN
        DEALLOCATE ( Hru_down, Hru_down_frac, Hru_down_fracwt )
        DEALLOCATE ( Cascade_area)
      ENDIF
      IF ( Cascadegw_flag>CASCADEGW_OFF ) THEN
        DEALLOCATE ( Gwr_down, Gwr_down_frac, Cascade_gwr_area )
!       DEALLOCATE ( Gwr_down_fracwt )
      ENDIF

      cascclean = 0
      END FUNCTION cascclean

!***********************************************************************
! Initialize cascading flow variables
!***********************************************************************
      SUBROUTINE init_cascade(Iret)
      USE PRMS_CONSTANTS, ONLY: DEBUG_less, INACTIVE, LAKE, SWALE, CASCADE_HRU_SEGMENT, CASCADE_NORMAL
      USE PRMS_MODULE, ONLY: Nhru, Nsegment, Ncascade, Ncascdgw, Print_debug, Cascade_flag, Cascadegw_flag
      USE PRMS_CASCADE
      USE PRMS_BASIN, ONLY: Active_hrus, Hru_route_order, Hru_type, Hru_area
      IMPLICIT NONE
! Functions
      INTEGER, EXTERNAL :: getparam
      EXTERNAL :: order_hrus, read_error
      INTRINSIC :: ABS
! Arguments
      INTEGER, INTENT(OUT) :: Iret
! Local Variables
      INTEGER :: i, j, k, ii, kk, dnhru, kup, jdn, istrm, num
      REAL, ALLOCATABLE :: hru_frac(:)
      REAL :: carea, frac
!***********************************************************************
      Iret = 0

! Cascade parameters
      Ndown = 1
      IF ( Cascade_flag==CASCADE_HRU_SEGMENT ) THEN
        ! simple 1 to 1 cascades, ncascade = nhru
        IF ( getparam(MODNAME, 'hru_segment', Nhru, 'integer', Hru_segment)/=0 ) CALL read_error(2, 'hru_segment')
        DO i = 1, Nhru
          Hru_up_id(i) = i
          Hru_strmseg_down_id(i) = Hru_segment(i)
        ENDDO
        Hru_down_id = 0
        Hru_pct_up = 1.0
        DEALLOCATE ( Hru_segment )
      ELSE
        IF ( getparam(MODNAME, 'hru_up_id', Ncascade, 'integer', Hru_up_id)/=0 ) CALL read_error(2, 'hru_up_id')
        IF ( getparam(MODNAME, 'hru_strmseg_down_id', Ncascade, 'integer', Hru_strmseg_down_id)/=0 ) &
     &       CALL read_error(2, 'hru_strmseg_down_id')
        IF ( getparam(MODNAME, 'hru_down_id', Ncascade, 'integer', Hru_down_id)/=0 ) CALL read_error(2, 'hru_down_id')
        IF ( getparam(MODNAME, 'hru_pct_up', Ncascade, 'real', Hru_pct_up)/=0 ) CALL read_error(2, 'hru_pct_up')
        ! figure out the maximum number of cascades links from all HRUs, to set dimensions for 2-D arrays
        Ncascade_hru = 0
        DO i = 1, Ncascade
          k = Hru_up_id(i)
          IF ( k>0 ) THEN
            jdn = Hru_down_id(i)
            Ncascade_hru(k) = Ncascade_hru(k) + 1
            IF ( Ncascade_hru(k)>Ndown ) Ndown = Ncascade_hru(k)
          ENDIF
        ENDDO
      ENDIF

      IF ( Cascadegw_flag==CASCADE_NORMAL ) THEN
        IF ( Cascade_flag==CASCADE_HRU_SEGMENT ) THEN
          Gw_up_id = Hru_up_id
          Gw_strmseg_down_id =  Hru_strmseg_down_id
          Gw_down_id = 0
          Gw_pct_up = 1.0
        ELSE
          IF ( getparam(MODNAME, 'gw_up_id', Ncascdgw, 'integer', Gw_up_id)/=0 ) CALL read_error(2, 'gw_up_id')
          IF ( getparam(MODNAME, 'gw_down_id', Ncascdgw, 'integer', Gw_down_id)/=0 ) CALL read_error(2, 'gw_down_id')
          IF ( getparam(MODNAME, 'gw_pct_up', Ncascdgw, 'real', Gw_pct_up)/=0 ) CALL read_error(2, 'gw_pct_up')
          IF ( getparam(MODNAME, 'gw_strmseg_down_id', Ncascdgw, 'integer', Gw_strmseg_down_id)/=0 ) &
     &         CALL read_error(2, 'gw_strmseg_down_id')
          Ncascade_gwr = 0
          DO i = 1, Ncascdgw
            k = Gw_up_id(i)
            IF ( k>0 ) THEN
              jdn = Gw_down_id(i)
              Ncascade_gwr(k) = Ncascade_gwr(k) + 1
              IF ( Ncascade_gwr(k)>Ndown ) Ndown = Ncascade_gwr(k)
            ENDIF
          ENDDO
        ENDIF
      ENDIF

! ??rsr, why 15??
      IF ( Ndown>15 .AND. Print_debug==13 ) WRITE ( MSGUNT, * ) 'possible ndown issue', Ndown

! allocate HRU variables
      ALLOCATE ( Hru_down(Ndown,Nhru), Cascade_area(Ndown,Nhru) )
      ALLOCATE ( Hru_down_frac(Ndown,Nhru), Hru_down_fracwt(Ndown,Nhru) )
      ALLOCATE ( hru_frac(Nhru) )
      Hru_down = 0
      Hru_down_frac = 0.0
      Hru_down_fracwt = 0.0
      Cascade_area = 0.0
      Ncascade_hru = 0
      hru_frac = 0.0

      DO i = 1, Ncascade
        kup = Hru_up_id(i)
        IF ( kup<1 ) THEN
          PRINT *, 'Cascade ignored as hru_up_id<1, cascade:', i, ', hru_up_id:', kup
          CYCLE
        ENDIF
        jdn = Hru_down_id(i)
        frac = Hru_pct_up(i)
        IF ( frac>0.9998 ) frac = 1.0
        istrm = Hru_strmseg_down_id(i)

        IF ( frac<0.00001 ) THEN
          IF ( Print_debug==13 ) WRITE (MSGUNT, 9004) 'Cascade ignored as hru_pct_up=0.0', &
     &                                                i, kup, jdn, frac, istrm
        ELSEIF ( istrm>Nsegment ) THEN
          IF ( Print_debug==13 ) WRITE (MSGUNT, 9004) 'Cascade ignored as segment>nsegment', &
     &                                                i, kup, jdn, frac, istrm
        ELSEIF ( kup<1 .AND. jdn==0 ) THEN
          IF ( Print_debug==13 ) WRITE (MSGUNT, 9004) 'Cascade ignored as up and down HRU = 0', &
     &                                                i, kup, jdn, frac, istrm
        ELSEIF ( istrm==0 .AND. jdn==0 ) THEN
          IF ( Print_debug==13 ) WRITE ( MSGUNT, 9004 ) 'Cascade ignored as down HRU and segment = 0', &
     &                                                  i, kup, jdn, frac, istrm
        ELSEIF ( Hru_type(kup)==INACTIVE ) THEN
          IF ( Print_debug==13 ) WRITE (MSGUNT, 9004) 'Cascade ignored as up HRU is inactive', &
     &                                                i, kup, jdn, frac, istrm
        ELSEIF ( Hru_type(kup)==SWALE ) THEN
          IF ( Print_debug==13 ) WRITE (MSGUNT, 9004) 'Cascade ignored as up HRU is a swale', &
     &                                                i, kup, jdn, frac, istrm
        ELSEIF ( Hru_type(kup)==LAKE .AND. istrm<1 ) THEN
          IF ( Print_debug==13 ) WRITE ( MSGUNT, 9004 ) 'Cascade ignored as lake HRU cannot cascade to an HRU', &
     &                                                  i, kup, jdn, frac, istrm
        ELSE
          IF ( jdn>0 .AND. istrm<1 ) THEN
            IF ( Hru_type(jdn)==INACTIVE ) THEN
              IF ( Print_debug==13 ) WRITE ( MSGUNT, 9004 ) &
     &             'Cascade ignored as down HRU is inactive', i, kup, jdn, frac, istrm
              CYCLE
            ENDIF
          ENDIF

          carea = frac*Hru_area(kup)
          ! get rid of small cascades, redistribute fractions
          IF ( carea<Cascade_tol .AND. frac<0.075 ) THEN
            IF ( Print_debug==13 ) WRITE ( MSGUNT, 9005 ) i, kup, jdn, frac*100.0, carea
          ELSEIF ( Cascade_flg==1 ) THEN
            IF ( frac>hru_frac(kup) ) THEN
              hru_frac(kup) = frac
              Ncascade_hru(kup) = 1
              Hru_down_frac(1, kup) = frac
              IF ( istrm>0 ) THEN
                Hru_down(1, kup) = -istrm
              ELSE
                Hru_down(1, kup) = jdn
              ENDIF
            ENDIF
          ELSE
            hru_frac(kup) = hru_frac(kup) + frac
            IF ( hru_frac(kup)>1.0 ) THEN
              IF ( hru_frac(kup)>1.00001 ) THEN
                IF ( Print_debug==13 ) WRITE ( MSGUNT, 9004 ) &
     &               'Addition of cascade link makes contributing area add up to > 1.0, thus fraction reduced', &
     &               i, kup, jdn, hru_frac(kup), istrm
              ENDIF
              frac = frac + 1.0 - hru_frac(kup)
              hru_frac(kup) = 1.0
            ENDIF
            Ncascade_hru(kup) = Ncascade_hru(kup) + 1
            kk = Ncascade_hru(kup)
            Hru_down_frac(kk, kup) = frac
            IF ( istrm>0 ) THEN
              Hru_down(kk, kup) = -istrm
            ELSE
              Hru_down(kk, kup) = jdn
            ENDIF
          ENDIF
        ENDIF
      ENDDO

      ! how do we route headwater HRUs to stream segment rather than
      ! across valleys**********************RSR???

      DO ii = 1, Active_hrus
        i = Hru_route_order(ii)
        num = Ncascade_hru(i)
        IF ( num==0 ) CYCLE
        DO k = 1, num
          frac = Hru_down_frac(k, i)
          Hru_down_frac(k,i) = frac + frac*(1.0-hru_frac(i))/hru_frac(i)
        ENDDO

        k = 1
        DO kk = 1, num
          dnhru = Hru_down(kk, i)
          IF ( dnhru==0 ) CYCLE
          Hru_down_frac(k, i) = Hru_down_frac(kk, i)
          Hru_down(k, i) = dnhru
          j = num
          DO WHILE (j>kk )
            IF ( dnhru==Hru_down(j, i) ) THEN
              Hru_down(j, i) = 0
              Hru_down_frac(k, i) = Hru_down_frac(k, i) + Hru_down_frac(j, i)
              IF ( Hru_down_frac(k, i)>1.00001 ) THEN
                IF ( Print_debug==13 ) THEN
                  WRITE ( MSGUNT, * ) 'Combining cascade links makes contributing area add up to > 1.0, thus fraction reduced.'
                  WRITE ( MSGUNT, * ) 'Up HRU:', i, ' Down HRU:', dnhru
                ENDIF
                Hru_down_frac(k, i) = 1.0
              ENDIF
              IF ( dnhru<0 ) THEN
! two cascades to same stream segment, combine
!                IF ( Print_debug>DEBUG_less ) PRINT 9002, i, 'stream segment', ABS(dnhru)
                IF ( Print_debug==13 ) WRITE ( MSGUNT, 9002 ) i, 'stream segment', ABS( dnhru )
              ELSE
! two cascades to same HRU, combine
                IF ( Print_debug==13 ) WRITE ( MSGUNT, 9002 ) i, 'downslope HRU', dnhru
              ENDIF
              Ncascade_hru(i) = Ncascade_hru(i) - 1
            ENDIF
            j = j - 1
          ENDDO
          Cascade_area(k, i) = Hru_down_frac(k, i)*Hru_area(i)
          IF ( dnhru>0 ) Hru_down_fracwt(k, i) = Cascade_area(k, i)/Hru_area(dnhru)
          k = k + 1
        ENDDO
      ENDDO
      DEALLOCATE ( hru_frac )

      CALL order_hrus(Iret)

      IF ( Print_debug==13 ) THEN
        WRITE ( MSGUNT, 9001 )
        WRITE ( MSGUNT, 9003 ) (Hru_route_order(i), i=1, Iorder)
      ENDIF

      DEALLOCATE ( Hru_up_id, Hru_pct_up, Hru_down_id, Hru_strmseg_down_id )

 9001 FORMAT (/, 'HRU routing order:')
 9002 FORMAT ('*** WARNING, combined multiple cascade paths from HRU:', I7, ' to ', A, ':', I7)
 9003 FORMAT (10I7)
 9004 FORMAT ('*** WARNING, ', A, /, '    Cascade:', I7, '; up HRU:', &
     &        I7, '; down HRU:', I7, '; up fraction:', F8.4, '; stream segment:', I5)
 9005 FORMAT ('*** WARNING, ignoring small cascade, carea<cascade_tol', &
     &        /, '    Cascade:', I7, '; HRU up:', I7, '; HRU down:', I7, &
     &        '; fraction up:', F8.2, '; cascade area:', F8.2)

      END SUBROUTINE init_cascade

!***********************************************************************
! order hrus allowing many to 1
!***********************************************************************
      SUBROUTINE order_hrus(Iret)
      USE PRMS_CONSTANTS, ONLY: ACTIVE, INACTIVE, LAND, LAKE, SWALE, GLACIER
      USE PRMS_CASCADE, ONLY: Hru_down, Iorder, MSGUNT, Circle_switch, Ncascade_hru
      USE PRMS_MODULE, ONLY: Nhru, Print_debug
      USE PRMS_BASIN, ONLY: Active_hrus, Hru_route_order, Hru_type
      IMPLICIT NONE
! Functions
      EXTERNAL :: up_tree, PRMS_open_module_file
!     Arguments
      INTEGER, INTENT(OUT) :: Iret
!     Local Variables
      INTEGER, ALLOCATABLE :: roots(:), path(:), hrus_up_list(:, :)
      INTEGER, ALLOCATABLE :: is_hru_on_list(:), dn_id_count(:)
      INTEGER, ALLOCATABLE :: up_id_count(:), up_id_cnt(:)
      INTEGER :: i, j, k, ii, nroots, circle_flg, ihru, npath, type_flag
      INTEGER :: goes_on_list, up_hru_id, added, dnhru, max_up_id_count, ounit
!-----------------------------------------------------------------------
!     up_id_count equals number of upslope HRUs an HRU has
!     dn_id_count equals number of downslope HRUs an HRU has
!     ncascade_hru equals number of downslope HRUs and stream segments an HRU has
      max_up_id_count = 0
      ALLOCATE (up_id_count(Nhru), dn_id_count(Nhru), roots(Nhru))
      ALLOCATE (path(Nhru), is_hru_on_list(Nhru))
      DO i = 1, Nhru
        up_id_count(i) = 0
        dn_id_count(i) = 0
        roots(i) = 0
        path(i) = 0
        is_hru_on_list(i) = 0
      ENDDO
      DO ii = 1, Active_hrus
        i = Hru_route_order(ii)
        DO k = 1, Ncascade_hru(i)
          dnhru = Hru_down(k, i)
          IF ( dnhru>0 ) THEN
            dn_id_count(i) = dn_id_count(i) + 1
            up_id_count(dnhru) = up_id_count(dnhru) + 1
!           determine the maximum up_id_count
            IF ( up_id_count(dnhru)>max_up_id_count ) max_up_id_count = up_id_count(dnhru)
          ENDIF
        ENDDO
      ENDDO
      ALLOCATE ( hrus_up_list(max_up_id_count, Nhru) )
      hrus_up_list = 0

! get the list of HRUs upslope of each HRU and root HRUs
      ALLOCATE ( up_id_cnt(Nhru) )
      nroots = 0
      type_flag = 0
      up_id_cnt = up_id_count
      DO ii = 1, Active_hrus
        i = Hru_route_order(ii)
        IF ( dn_id_count(i)==0 ) THEN
          nroots = nroots + 1
          roots(nroots) = i
        ENDIF
        IF ( up_id_count(i)==0 ) THEN
          !HRU does not receive or cascade flow - swale
          IF ( (Hru_type(i)==LAND.OR.Hru_type(i)==GLACIER) .AND. Ncascade_hru(i)==0 ) THEN
            IF ( Print_debug==13 ) WRITE ( MSGUNT, 9008 ) i
            PRINT 9008, i
            Hru_type(i) = SWALE
            type_flag = 1
            CYCLE
          ENDIF
        ENDIF
        IF ( (Hru_type(i)==LAND.OR.Hru_type(i)==GLACIER) .AND. Ncascade_hru(i)==0 ) THEN
          !HRU does not cascade flow - swale
          IF ( Print_debug==13 ) WRITE ( MSGUNT, 9009 ) i
          PRINT 9009, i
          Hru_type(i) = SWALE
          type_flag = 1
          CYCLE
        ELSE
          DO k = 1, Ncascade_hru(i)
            dnhru = Hru_down(k, i)
            IF ( dnhru>0 ) THEN
              hrus_up_list(up_id_cnt(dnhru), dnhru) = i
              up_id_cnt(dnhru) = up_id_cnt(dnhru) - 1
            ENDIF
          ENDDO
        ENDIF
      ENDDO
      DEALLOCATE ( up_id_cnt )
      IF ( type_flag==1 ) THEN
        PRINT *, 'hru_type parameter written to file hru_typexxx.param'
        PRINT *, 'can use to replace current values to address warnings'
        CALL PRMS_open_module_file(ounit, 'hru_typexxx.param')
        WRITE ( ounit, 11 ) Nhru
        WRITE ( ounit, '(I1)' ) ( Hru_type(i), i = 1, Nhru )
        CLOSE ( ounit )
   11   FORMAT ('####', /, 'hru_type', /, '1', /, 'nhru', /, I7, /, '1')
      ENDIF

      Iret = 0
! check for circles when circle_switch = 1
      IF ( Circle_switch==ACTIVE ) THEN
        circle_flg = 0
        DO i = 1, nroots
          ihru = roots(i)
          path(1) = ihru
          npath = 1
          circle_flg = 0
          CALL up_tree(Nhru, ihru, up_id_count, hrus_up_list, npath, path, circle_flg, max_up_id_count)
          IF ( circle_flg==1 ) Iret = 1
        ENDDO
        IF ( circle_flg==1 ) THEN
          PRINT 9005
          IF ( Print_debug==13 ) WRITE ( MSGUNT, 9005 )
          Iret = 1
          RETURN
        ENDIF
      ENDIF
      DEALLOCATE ( path )

! Determine HRU routing order
      Hru_route_order = 0
      Iorder = 0  !number of HRUs added to Hru_route_order
      DO WHILE ( Iorder < Active_hrus )
        added = 0
        DO i = 1, Nhru
          IF ( Hru_type(i)==INACTIVE ) CYCLE !ignore inactive HRUs
          IF ( is_hru_on_list(i)==0 ) THEN
            goes_on_list = 1
            DO j = 1, up_id_count(i)
              up_hru_id = hrus_up_list(j, i)
              ! if upslope HRU not on list, can't add HRU i
              IF ( is_hru_on_list(up_hru_id)==0 ) THEN
                goes_on_list = 0
                EXIT
              ENDIF
            ENDDO
            !add HRU to list
            IF ( goes_on_list==1 ) THEN
              is_hru_on_list(i) = 1
              Iorder = Iorder + 1
              Hru_route_order(Iorder) = i
              added = 1
            ENDIF
          ENDIF
        ENDDO
        IF ( added==0 ) THEN
          PRINT *, 'ERROR, no HRUs added to routing order on last pass through cascades, possible circles'
          DO i = 1, Nhru
            IF ( is_hru_on_list(i)==0 ) THEN
              PRINT *, 'HRU not in order:', i
              IF ( Print_debug==13 ) WRITE ( MSGUNT, * ) 'HRU not in order:', i
            ENDIF
          ENDDO
          PRINT 9002, (Hru_route_order(i), i=1, Iorder)
          IF ( Print_debug==13 ) THEN
            WRITE (MSGUNT, *) 'ERROR, no HRUs added to routing order on last pass through cascades, possible circles'
            WRITE ( MSGUNT, 9001 ) Iorder
            WRITE ( MSGUNT, 9002 ) (Hru_route_order(i), i=1, Iorder)
          ENDIF
          Iret = 1
          RETURN
        ENDIF
      ENDDO
      DEALLOCATE ( hrus_up_list, up_id_count, dn_id_count )

      IF ( Print_debug==13 ) THEN
        WRITE ( MSGUNT, 9003 ) nroots
        WRITE ( MSGUNT, 9002 ) (roots(i), i=1, nroots)
      ENDIF
      DEALLOCATE ( roots )

      IF ( Iorder/=Active_hrus ) THEN
        PRINT 9004, Iorder, Nhru, Active_hrus
        IF ( Print_debug==13 ) WRITE ( MSGUNT, 9004 ) Iorder, Nhru, Active_hrus
        DO i = 1, Nhru
          IF ( is_hru_on_list(i)==0 ) THEN
            IF ( Hru_type(i)/=INACTIVE ) THEN
              PRINT 9006, i
              IF ( Print_debug==13 ) WRITE ( MSGUNT, 9006 ) i
              Iret = 1
            ELSE
              IF ( Print_debug==13 ) WRITE ( MSGUNT, 9007 ) i
            ENDIF
          ENDIF
        ENDDO
      ENDIF
      DEALLOCATE ( is_hru_on_list )

 9001 FORMAT (/, I7, ' HRUs in routing order)')
 9002 FORMAT (12I7)
 9003 FORMAT (/, I6, ' HRUs that do not cascade to another HRU (roots)')
 9004 FORMAT (/, 'WARNING, not all HRUs are included in the cascading pattern, likely circle or inactive HRUs', //, &
     &        'Number of HRUs in pattern:', I7, ', number of HRUs:', I7, &
     &        ', Active HRUs:', I7, //, 'HRUs not in routing order:')
 9005 FORMAT (/, 'ERROR, circular HRU path found', /)
 9006 FORMAT (I7, ' missing')
 9007 FORMAT (I7, ' inactive')
 9008 FORMAT ('WARNING, HRU', I7, ' does not cascade or receive flow', /, &
     &        9X, 'and was specified as hru_type = 1,', /, 9X, &
     &        'hru_type was changed to 3 (swale)', /)
 9009 FORMAT ('WARNING, HRU', I7, ' receives flow but does not cascade', &
     &        /, 9X, 'and was specified as hru_type 1,', /, 9X, &
     &        'hru_type was changed to 3 (swale)', /)

      END SUBROUTINE order_hrus

!***********************************************************************
! Initialize cascading flow variables
!***********************************************************************
      SUBROUTINE initgw_cascade(Iret)
      USE PRMS_CONSTANTS, ONLY: DEBUG_less
      USE PRMS_MODULE, ONLY: Ngw, Nsegment, Ncascdgw, Print_debug, Gwr_swale_flag
      USE PRMS_CASCADE
      USE PRMS_BASIN, ONLY: Active_gwrs, Gwr_route_order, Gwr_type, Hru_area
      IMPLICIT NONE
! Functions
      EXTERNAL :: order_gwrs
      INTRINSIC :: ABS, DBLE
! Arguments
      INTEGER, INTENT(OUT) :: Iret
! Local Variables
      INTEGER :: i, j, k, ii, kk, dngwr, kup, jdn, istrm, num
      REAL :: carea, frac
      REAL, ALLOCATABLE :: gwr_frac(:)
!***********************************************************************
      Iret = 0

      ALLOCATE ( gwr_frac(Ngw) )
      DO i = 1, Ngw
        DO k = 1, Ndown
          Gwr_down(k, i) = 0
          Gwr_down_frac(k, i) = 0.0
!         Gwr_down_fracwt(k, i) = 0.0
          Cascade_gwr_area(k, i) = 0.0
        ENDDO
        gwr_frac(i) = 0.0
        Ncascade_gwr(i) = 0
      ENDDO

      IF ( Print_debug==13 ) WRITE ( MSGUNT, * )

      DO i = 1, Ncascdgw
        kup = Gw_up_id(i)
        IF ( kup<1 ) THEN
          PRINT *, 'Cascade ignored as gw_up_id<1, cascade:', i, kup
          CYCLE
        ENDIF
        jdn = Gw_down_id(i)
        frac = Gw_pct_up(i)
        IF ( frac>1.0 ) frac = 1.0
        istrm = Gw_strmseg_down_id(i)

        IF ( frac<0.00001 ) THEN
          IF ( Print_debug==13 ) WRITE ( MSGUNT, 9004 ) 'Cascade ignored as gw_pct_up=0.0', &
     &                                                  i, kup, jdn, frac, istrm
        ELSEIF ( istrm>Nsegment ) THEN
          IF ( Print_debug==13 ) WRITE (MSGUNT, 9004) 'Cascade ignored as segment>nsegment', &
     &                                                i, kup, jdn, frac, istrm
        ELSEIF ( kup<1 .AND. jdn==0 ) THEN
          IF ( Print_debug==13 ) WRITE (MSGUNT, 9004) 'Cascade ignored as up and down GWR = 0', &
     &                                                i, kup, jdn, frac, istrm
        ELSEIF ( istrm==0 .AND. jdn==0 ) THEN
          IF ( Print_debug==13 ) WRITE ( MSGUNT, 9004 ) 'Cascade ignored as down GWR and segment = 0', &
     &                                                  i, kup, jdn, frac, istrm
        ELSEIF ( Gwr_type(kup)==0 ) THEN
          IF ( Print_debug==13 ) WRITE (MSGUNT, 9004) 'Cascade ignored as up GWR is inactive', &
     &                                                i, kup, jdn, frac, istrm
        ELSEIF ( Gwr_type(kup)==3 ) THEN
          IF ( Gwr_swale_flag>0 ) THEN
            IF ( Print_debug==13 ) WRITE ( MSGUNT, 9004 ) 'Cascade ignored as up GWR is a swale', &
     &                                                    i, kup, jdn, frac, istrm
          ELSE
            PRINT 9004, 'ERROR, Cascade as up GWR specified as a swale and gwr_swale_flag = 0', &
     &                  i, kup, jdn, frac, istrm
          ENDIF
        ELSE
          IF ( jdn>0 .AND. istrm<1 ) THEN
            IF ( Gwr_type(jdn)==0 ) THEN
              IF ( Print_debug==13 ) WRITE ( MSGUNT, 9004 ) 'Cascade ignored as down GWR is inactive', &
     &                                                      i, kup, jdn, frac, istrm
              CYCLE
            ENDIF
          ENDIF

          ! want to ignore streams within lakes, not sure how to do yet
!         IF ( jdn>0 ) THEN
!           IF ( Gwr_type(jdn)==2 .AND. istrm>1 ) THEN
!             istrm = 0
!             IF ( Print_debug==13 ) WRITE (MSGUNT, 9004) &
!    &             'Up GWR of a lake GWR cannot cascade to stream segment, gw_strmseg_down_id set to 0', &
!    &             i, kup, jdn, frac, istrm
!           ENDIF
!         ENDIF

          carea = frac*Hru_area(kup)
          ! get rid of small cascades, redistribute fractions
          IF ( carea<Cascade_tol .AND. frac<0.075 ) THEN
            IF ( Print_debug==13 ) WRITE ( MSGUNT, 9005 ) i, kup, jdn, frac*100.0, carea
          ELSEIF ( Cascade_flg==1 ) THEN
            IF ( frac>gwr_frac(kup) ) THEN
              gwr_frac(kup) = frac
              Ncascade_gwr(kup) = 1
              Gwr_down_frac(1, kup) = frac
              IF ( istrm>0 ) THEN
                Gwr_down(1, kup) = -istrm
              ELSE
                Gwr_down(1, kup) = jdn
              ENDIF
            ENDIF
          ELSE
            gwr_frac(kup) = gwr_frac(kup) + frac
            IF ( gwr_frac(kup)>1.0 ) THEN
              IF ( gwr_frac(kup)>1.00001 ) THEN
                IF ( Print_debug==13 ) WRITE ( MSGUNT, 9004 ) &
     &               'Addition of GWR cascade link makes contributing area add up to > 1.0, thus fraction reduced', &
     &               i, kup, jdn, gwr_frac(kup), istrm
              ENDIF
              frac = frac + 1.0 - gwr_frac(kup)
              gwr_frac(kup) = 1.0
            ENDIF
            Ncascade_gwr(kup) = Ncascade_gwr(kup) + 1
            kk = Ncascade_gwr(kup)
            Gwr_down_frac(kk, kup) = frac
            IF ( istrm>0 ) THEN
              Gwr_down(kk, kup) = -istrm
            ELSE
              Gwr_down(kk, kup) = jdn
            ENDIF
          ENDIF
        ENDIF
      ENDDO

      DO ii = 1, Active_gwrs
        i = Gwr_route_order(ii)
        num = Ncascade_gwr(i)
        IF ( num==0 ) CYCLE
        DO k = 1, num
          frac = Gwr_down_frac(k, i)
          Gwr_down_frac(k,i) = frac + frac*(1.0-gwr_frac(i))/gwr_frac(i)
        ENDDO

        k = 1
        DO kk = 1, num
          dngwr = Gwr_down(kk, i)
          IF ( dngwr==0 ) CYCLE
          Gwr_down_frac(k, i) = Gwr_down_frac(kk, i)
          Gwr_down(k, i) = dngwr
          j = num
          DO WHILE (j>kk )
            IF ( dngwr==Gwr_down(j, i) ) THEN
              Gwr_down(j, i) = 0
              Gwr_down_frac(k, i) = Gwr_down_frac(k, i) + Gwr_down_frac(j, i)
              IF ( Gwr_down_frac(k, i)>1.00001 ) THEN
                IF ( Print_debug==13 ) THEN
                  WRITE ( MSGUNT, * ) 'Combining cascade links makes contributing area add up to > 1.0, thus fraction reduced'
                  WRITE ( MSGUNT,* ) 'Up GWR:', i, ' Down GWR:', dngwr
                ENDIF
                Gwr_down_frac(k, i) = 1.0
              ENDIF
              IF ( dngwr<0 ) THEN
! two cascades to same stream segment, combine
!                IF ( Print_debug>DEBUG_less ) PRINT 9002, i, 'stream segment', ABS(dngwr)
                IF ( Print_debug==13 ) WRITE ( MSGUNT, 9002 ) i, 'stream segment', ABS( dngwr )
              ELSE
! two cascades to same HRU, combine
                IF ( Print_debug==13 ) WRITE ( MSGUNT, 9002 ) i, 'downslope GWR', dngwr
              ENDIF
              Ncascade_gwr(i) = Ncascade_gwr(i) - 1
            ENDIF
            j = j - 1
          ENDDO
          Cascade_gwr_area(k, i) = Gwr_down_frac(k, i)*Hru_area(i)
!          IF ( dngwr>0 ) Gwr_down_fracwt(k, i) = Cascade_gwr_area(k, i)/Hru_area(dnhru)
          k = k + 1
        ENDDO
      ENDDO
      DEALLOCATE ( gwr_frac )

      CALL order_gwrs(Iret)

      IF ( Print_debug==13 ) THEN
        WRITE ( MSGUNT, 9001 )
        WRITE ( MSGUNT, 9003 ) (Gwr_route_order(i), i=1, Igworder)
      ENDIF

      DEALLOCATE ( Gw_strmseg_down_id, Gw_up_id, Gw_down_id, Gw_pct_up )

 9001 FORMAT (/, 'GWR routing order:')
 9002 FORMAT ('WARNING, combined multiple cascade paths from GWR:', I7, ' to ', A, ':', I7)
 9003 FORMAT (10I7)
 9004 FORMAT ('*** WARNING, ', A, /, '    Cascade:', I7, '; up GWR:', &
     &        I7, '; down GWR:', I7, '; up fraction:', F8.4, '; stream segment:', I5)
 9005 FORMAT ('*** WARNING, ignoring small cascade, carea<cascade_tol', &
     &        /, '    Cascade:', I7, '; GWR up:', I7, '; GWR down:', I7, &
     &        '; fraction up:', F8.2, '; cascade area:', F8.2)

      END SUBROUTINE initgw_cascade

!***********************************************************************
! order GWRs allowing many to 1
!***********************************************************************
      SUBROUTINE order_gwrs(Iret)
      USE PRMS_MODULE, ONLY: Ngw, Print_debug, Gwr_swale_flag
      USE PRMS_CASCADE, ONLY: Gwr_down, Igworder, MSGUNT, Circle_switch, Ncascade_gwr
      USE PRMS_BASIN, ONLY: Active_gwrs, Gwr_route_order, Gwr_type
      IMPLICIT NONE
! Functions
      EXTERNAL :: up_tree
!     Arguments
      INTEGER, INTENT(OUT) :: Iret
!     Local Variables
      INTEGER, ALLOCATABLE :: roots(:), path(:), gwrs_up_list(:, :)
      INTEGER, ALLOCATABLE :: is_gwr_on_list(:), dn_id_count(:)
      INTEGER, ALLOCATABLE :: up_id_count(:), up_id_cnt(:)
      INTEGER :: i, j, k, ii, nroots, circle_flg, igwr, npath, swalegwr
      INTEGER :: goes_on_list, up_gwr_id, added, dngwr, max_up_id_count, ounit
!-----------------------------------------------------------------------
!     up_id_count equals number of upslope GWRs a GWR has
!     dn_id_count equals number of downslope GWRs a GWR has
!     ncascade_gwr equals number of downslope GWRs and stream segments a GWR has
      max_up_id_count = 0
      ALLOCATE ( up_id_count(Ngw), dn_id_count(Ngw), roots(Ngw) )
      ALLOCATE ( path(Ngw), is_gwr_on_list(Ngw) )
      up_id_count = 0
      dn_id_count = 0
      roots = 0
      path = 0
      is_gwr_on_list = 0
      DO ii = 1, Active_gwrs
        i = Gwr_route_order(ii)
        DO k = 1, Ncascade_gwr(i)
          dngwr = Gwr_down(k, i)
          IF ( dngwr>0 ) THEN
            dn_id_count(i) = dn_id_count(i) + 1
            up_id_count(dngwr) = up_id_count(dngwr) + 1
!           determine the maximum up_id_count
            IF ( up_id_count(dngwr)>max_up_id_count ) max_up_id_count = up_id_count(dngwr)
          ENDIF
        ENDDO
      ENDDO
      ALLOCATE ( gwrs_up_list(max_up_id_count, Ngw) )
      gwrs_up_list = 0

! get the list of GWRs upslope of each GWR and root GWRs
      ALLOCATE ( up_id_cnt(Ngw) )
      nroots = 0
      up_id_cnt = up_id_count
      swalegwr = 0 ! allow swales if gwr_swale_flag > 0
      DO ii = 1, Active_gwrs
        i = Gwr_route_order(ii)
        IF ( dn_id_count(i)==0 ) THEN
          nroots = nroots + 1
          roots(nroots) = i
        ENDIF
        IF ( up_id_count(i)==0 ) THEN
          IF ( Ncascade_gwr(i)==0 ) THEN
            !GWR does not receive or cascade flow - swale
            IF ( Gwr_swale_flag>0 ) THEN
              IF ( Gwr_type(i)==1 ) THEN
                PRINT 9008, i
                IF ( Print_debug==13 ) WRITE ( MSGUNT, 9008 ) i
                Gwr_type(i) = 3
                swalegwr = 2
              ENDIF
            ELSE
              PRINT 9010, i
              IF ( Print_debug==13 ) WRITE ( MSGUNT, 9010 ) i
              swalegwr = 1
              CYCLE
            ENDIF
          ENDIF
        ENDIF
        IF ( Ncascade_gwr(i)==0 ) THEN
          !GWR does not cascade flow
          IF ( Gwr_swale_flag>0 ) THEN
            IF ( Gwr_type(i)==1 ) THEN
              PRINT 9009, i
              IF ( Print_debug==13 ) WRITE ( MSGUNT, 9009 ) i
              Gwr_type(i) = 3
              swalegwr = 2
            ENDIF
          ELSE
            PRINT 9011, i
            IF ( Print_debug==13 ) WRITE ( MSGUNT, 9011 ) i
            swalegwr = 1
          ENDIF
        ELSE
          DO k = 1, Ncascade_gwr(i)
            dngwr = Gwr_down(k, i)
            IF ( dngwr>0 ) THEN
              gwrs_up_list(up_id_cnt(dngwr), dngwr) = i
              up_id_cnt(dngwr) = up_id_cnt(dngwr) - 1
            ENDIF
          ENDDO
        ENDIF
      ENDDO
      IF ( swalegwr==1 ) THEN
        Iret = 1
        RETURN
      ENDIF
      DEALLOCATE ( up_id_cnt )

      IF ( swalegwr==2 ) THEN
        PRINT *, 'gwr_type parameter written to file gwr_typexxx.param'
        PRINT *, 'not used yet, but written for future use to address warnings'
        CALL PRMS_open_module_file(ounit, 'gwr_typexxx.param')
        WRITE ( ounit, 11 ) Ngw
        WRITE ( ounit, '(I1)' ) ( Gwr_type(i), i = 1, Ngw )
        CLOSE ( ounit )
  11    FORMAT ('####', /, 'gwr_type', /, '1', /, 'ngw', /, I7, /, '2')
      ENDIF

      Iret = 0
! check for circles when circle_switch = 1
      IF ( Circle_switch==1 ) THEN
        circle_flg = 0
        DO i = 1, nroots
          igwr = roots(i)
          path(1) = igwr
          npath = 1
          circle_flg = 0
          CALL up_tree(Ngw, igwr, up_id_count, gwrs_up_list, npath, path, circle_flg, max_up_id_count)
          IF ( circle_flg==1 ) Iret = 1
        ENDDO
        IF ( circle_flg==1 ) THEN
          PRINT 9005
          IF ( Print_debug==13 ) WRITE ( MSGUNT, 9005 )
          Iret = 1
          RETURN
        ENDIF
      ENDIF
      DEALLOCATE ( path )

! Determine GWR routing order
      Gwr_route_order = 0
      Igworder = 0  !number of GWRs added to Gwr_route_order
      DO WHILE ( Igworder < Active_gwrs )
        added = 0
        DO i = 1, Ngw
          IF ( Gwr_type(i)==0 ) CYCLE !ignore inactive GWRs
          IF ( is_gwr_on_list(i)==0 ) THEN
            goes_on_list = 1
            DO j = 1, up_id_count(i)
              up_gwr_id = gwrs_up_list(j, i)
              ! if upslope GWR not on list, can't add GWR i
              IF ( is_gwr_on_list(up_gwr_id)==0 ) THEN
                goes_on_list = 0
                EXIT
              ENDIF
            ENDDO
            !add GWR to list
            IF ( goes_on_list==1 ) THEN
              is_gwr_on_list(i) = 1
              Igworder = Igworder + 1
              Gwr_route_order(Igworder) = i
              added = 1
            ENDIF
          ENDIF
        ENDDO
        IF ( added==0 ) THEN
          PRINT *, 'ERROR, no GWRs added to routing order on last pass through cascades, possible circles'
          PRINT *, '       active GWRs:', Active_gwrs, '; number in routing order:', Igworder
          DO i = 1, Ngw
            IF ( is_gwr_on_list(i)==0 ) THEN
              PRINT *, 'GWR not in routing order', i
              IF ( Print_debug==13 ) WRITE ( MSGUNT, * ) 'GWR not in routing order', i
            ENDIF
          ENDDO
          PRINT 9002, (Gwr_route_order(i), i=1, Igworder)
          IF ( Print_debug==13 ) THEN
            WRITE ( MSGUNT, * ) 'ERROR, no GWRs added to routing order on last pass through cascades, possible circles'
            WRITE ( MSGUNT, * ) '       active GWRs:', Active_gwrs, '; number in routing order:', Igworder
            WRITE ( MSGUNT, 9001 ) Igworder
            WRITE ( MSGUNT, 9002 ) (Gwr_route_order(i), i=1, Igworder)
          ENDIF
          Iret = 1
          RETURN
        ENDIF
      ENDDO
      DEALLOCATE ( gwrs_up_list, up_id_count, dn_id_count )

      IF ( Print_debug==13 ) THEN
        WRITE ( MSGUNT, 9003 ) nroots
        WRITE ( MSGUNT, 9002 ) (roots(i), i=1, nroots)
      ENDIF
      DEALLOCATE ( roots )

      IF ( Igworder/=Active_gwrs ) THEN
        PRINT 9004, Igworder, Ngw, Active_gwrs
        IF ( Print_debug==13 ) WRITE ( MSGUNT, 9004 ) Igworder, Ngw, Active_gwrs
        DO i = 1, Ngw
          IF ( is_gwr_on_list(i)==0 ) THEN
            IF ( Gwr_type(i)/=0 ) THEN
              PRINT 9006, i
              IF ( Print_debug==13 ) WRITE ( MSGUNT, 9006 ) i
              Iret = 1
            ELSE
              IF ( Print_debug==13 ) WRITE ( MSGUNT, 9007 ) i
            ENDIF
          ENDIF
        ENDDO
      ENDIF
      DEALLOCATE ( is_gwr_on_list )

 9001 FORMAT (/, I6, ' GWRs that do not receive upslope flow (heads)')
 9002 FORMAT (10I7)
 9003 FORMAT (/, I6, ' GWRs that do not cascade to another GWR (roots)')
 9004 FORMAT (/, 'WARNING, not all GWRs are included in the cascading pattern, likely circle or inactive GWRs', //, &
     &        'Number of GWRs in pattern:', I7, ', number of GWRs:', I7, &
     &        ', Active GWRs:', I7, //, 'GWRs not in routing order:')
 9005 FORMAT (/, 'ERROR, circular GWR path found', /)
 9006 FORMAT (I7, ' missing')
 9007 FORMAT (I7, ' inactive')
 9008 FORMAT ('WARNING, GWR', I7, ' does not cascade or receive flow', /, &
     &        9X, 'and gwr_swale_flag > 0 and type = 1,', /, 9X, &
     &        'type changed to 3 (swale)', /)
 9009 FORMAT ('WARNING, GWR', I7, ' receives flow but does not cascade', /, &
     &        9X, 'and gwr_swale_flag > 0 and type = 1,', /, 9X, &
     &        'type changed to 3 (swale)', /)
 9010 FORMAT ('ERROR, GWR', I7, ' does not cascade or receive flow and gwr_swale_flag = 0')
 9011 FORMAT ('ERROR, GWR', I7, ' does not cascade flow and gwr_swale_flag = 0')

      END SUBROUTINE order_gwrs

!***********************************************************************
! Recursively walk up a tree of cascading spatial units
!***********************************************************************
      RECURSIVE SUBROUTINE up_tree(Num, N, Down_id, Up_list, Npath, Path, Circle_flg, Imx)
      IMPLICIT NONE
! Functions
      EXTERNAL :: check_path
! Arguments
      INTEGER, INTENT(IN) :: Num, N, Imx
      INTEGER, INTENT(IN) :: Down_id(Num), Up_list(Imx, Num)
      INTEGER, INTENT(INOUT) :: Npath, Path(Num), Circle_flg
! Local Variables
      INTEGER :: nup, i, parent
!-----------------------------------------------------------------------
      IF ( Circle_flg==1 ) RETURN
      nup = Down_id(N)
      DO i = 1, nup
        Npath = Npath + 1
        parent = Up_list(i, N)
        Path(Npath) = parent
        CALL check_path(Npath, Path, Circle_flg, nup)
        CALL up_tree(Num, parent, Down_id, Up_list, Npath, Path, Circle_flg, Imx)
      ENDDO

      IF ( nup==0 ) CALL check_path(Npath, Path, Circle_flg, nup)
      Npath = Npath - 1

      END SUBROUTINE up_tree

!***********************************************************************
! check for circular path
!***********************************************************************
      SUBROUTINE check_path(Npath, Path, Circle_flg, Nup)
      USE PRMS_MODULE, ONLY: Print_debug
      USE PRMS_CASCADE, ONLY: MSGUNT
      IMPLICIT NONE
!     Functions
      INTRINSIC :: MIN
!     Arguments
      INTEGER, INTENT(IN) :: Npath, Path(Npath), Nup
      INTEGER, INTENT(OUT) :: Circle_flg
!     Local Variables
      INTEGER :: j, i, n
!-----------------------------------------------------------------------
      Circle_flg = 0
      DO j = 1, Npath - 1
        DO i = j + 1, Npath
          IF ( Path(j)==Path(i) ) THEN
            PRINT *, 'ERROR, circular cascading path specified'
            PRINT *, Path
            Circle_flg = 1
          ENDIF
        ENDDO
      ENDDO

      IF ( Circle_flg==1 .OR. Nup==0 ) THEN
        IF ( Print_debug==13 ) THEN
          WRITE ( MSGUNT, * ) 'Cascade Path with', Npath, ' links'
          DO i = 1, Npath, 10
            n = MIN( Npath, i+9 )
            WRITE ( MSGUNT, 9001 ) (Path(j), j = i, n)
          ENDDO
        ENDIF
      ENDIF
 9001 FORMAT (10I8)

      END SUBROUTINE check_path
