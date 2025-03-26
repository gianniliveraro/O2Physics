// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
//  *+-+*+-+*+-+*+-+*+-+*+-+*
//  v0assocqa task
//  *+-+*+-+*+-+*+-+*+-+*+-+*
//
//  This task loops over a set of V0 indices and
//  checks for v0-to-collision association
//
//    Comments, questions, complaints, suggestions?
//    Please write to:
//    david.dobrigkeit.chinellato@cern.ch
//    gianni.shigeru.setoue.liveraro@cern.ch

#include <string>
#include <vector>
#include <cmath>
#include <array>
#include <cstdlib>
#include <map>
#include <iterator>
#include <utility>
#include <TVector2.h>

#include "TRandom3.h"
#include "Framework/runDataProcessing.h"
#include "Framework/RunningWorkflowInfo.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "Framework/ASoA.h"
#include "DCAFitter/DCAFitterN.h"
#include "ReconstructionDataFormats/Track.h"
#include "Common/Core/RecoDecay.h"
#include "Common/Core/trackUtilities.h"
#include "PWGLF/DataModel/LFStrangenessTables.h"
#include "PWGLF/DataModel/LFParticleIdentification.h"
#include "Common/Core/TrackSelection.h"
#include "Common/DataModel/TrackSelectionTables.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GeometryManager.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsCalibration/MeanVertexObject.h"
#include "TableHelper.h"
#include "Common/DataModel/EventSelection.h"
#include "Common/DataModel/Centrality.h"
#include "Common/DataModel/PIDResponse.h"
#include "Common/DataModel/Multiplicity.h"

#include "Common/CCDB/ctpRateFetcher.h"
#include "Common/Core/TPCVDriftManager.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;

// For MC association in pre-selection
using LabeledTracksExtra = soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksCov, aod::McTrackLabels>;

struct v0assoqa {

  Preslice<aod::V0s> perCollision = o2::aod::v0::collisionId;

  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  Configurable<std::string> irSource{"irSource", "T0VTX", "Estimator of the interaction rate (Recommended: pp --> T0VTX, Pb-Pb --> ZNC hadronic)"};
  
  Configurable<int> PDGCodePosDau{"PDGCodePosDau", -11, "select PDG code of positive daughter track"};
  Configurable<int> PDGCodeNegDau{"PDGCodeNegDau", 11, "select PDG code of negative daughter track"};
  Configurable<int> PDGCodeMother{"PDGCodeMother", 22, "select PDG code of mother particle"};

  Configurable<float> dcanegtopv{"dcanegtopv", 0.05, "DCA Neg To PV"};
  Configurable<float> dcapostopv{"dcapostopv", 0.05, "DCA Pos To PV"};
  Configurable<double> v0cospa{"v0cospa", 0.90, "V0 CosPA"}; // double -> N.B. dcos(x)/dx = 0 at x=0)
  Configurable<float> dcav0dau{"dcav0dau", 3.0, "DCA V0 Daughters"};
  Configurable<float> v0radius{"v0radius", 0.5, "v0radius"};

  // select momentum slice if desired
  Configurable<float> minimumPt{"minimumPt", 0.0f, "Minimum pT to store candidate"};
  Configurable<float> maximumPt{"maximumPt", 1000.0f, "Maximum pT to store candidate"};
  Configurable<float> maxDaughterEta{"maxDaughterEta", 5.0, "Maximum daughter eta"};

  Configurable<bool> doVDriftMgr{"doVDriftMgr", true, "Apply z-drift correction for photon-like v0s"};
  Configurable<bool> fGetIR{"fGetIR", false, "Flag to retrieve the IR info."};

  // CCDB options
  struct : ConfigurableGroup {
    Configurable<std::string> ccdburl{"ccdbConfigurations.ccdb-url", "http://alice-ccdb.cern.ch", "url of the ccdb repository"};
    Configurable<std::string> grpPath{"ccdbConfigurations.grpPath", "GLO/GRP/GRP", "Path of the grp file"};
    Configurable<std::string> grpmagPath{"ccdbConfigurations.grpmagPath", "GLO/Config/GRPMagField", "CCDB path of the GRPMagField object"};
    Configurable<std::string> lutPath{"ccdbConfigurations.lutPath", "GLO/Param/MatLUT", "Path of the Lut parametrization"};
    Configurable<std::string> geoPath{"ccdbConfigurations.geoPath", "GLO/Config/GeometryAligned", "Path of the geometry file"};
    Configurable<std::string> mVtxPath{"ccdbConfigurations.mVtxPath", "GLO/Calib/MeanVertex", "Path of the mean vertex file"};
  } ccdbConfigurations;

  // Operation and minimisation criteria
  struct : ConfigurableGroup {
    Configurable<double> d_bz_input{"dcaFitterConfigurations.d_bz", -999, "bz field, -999 is automatic"};
    Configurable<bool> d_UseAbsDCA{"dcaFitterConfigurations.d_UseAbsDCA", true, "Use Abs DCAs"};
    Configurable<bool> d_UseWeightedPCA{"dcaFitterConfigurations.d_UseWeightedPCA", false, "Vertices use cov matrices"};
    Configurable<bool> d_UseCollinearFit{"dcaFitterConfigurations.d_UseCollinearFit", false, "Fit V0s via the collinear Method in DCAFitter"};
    Configurable<float> d_maxDZIni{"dcaFitterConfigurations.d_maxDZIni", 1e9, "Dont consider a seed (circles intersection) if Z distance exceeds this"};
    Configurable<float> d_maxDXYIni{"dcaFitterConfigurations.d_maxDXYIni", 4, "Dont consider a seed (circles intersection) if XY distance exceeds this"};
    Configurable<int> useMatCorrType{"dcaFitterConfigurations.useMatCorrType", 2, "0: none, 1: TGeo, 2: LUT"};
    Configurable<int> rejDiffCollTracks{"dcaFitterConfigurations.rejDiffCollTracks", 0, "rejDiffCollTracks"};
  } dcaFitterConfigurations;

  int mRunNumber;
  float d_bz;
  float maxSnp;  // max sine phi for propagation
  float maxStep; // max step size (cm) for propagation
  Service<o2::ccdb::BasicCCDBManager> ccdb;
  o2::base::MatLayerCylSet* lut = nullptr;
  o2::base::Propagator::MatCorrType matCorr = o2::base::Propagator::MatCorrType::USEMatCorrNONE;
  o2::aod::common::TPCVDriftManager mVDriftMgr;

  o2::dataformats::MeanVertexObject* mVtx = nullptr;

  // Define o2 fitter, 2-prong, active memory (no need to redefine per event)
  o2::vertexing::DCAFitterN<2> fitter;

  // CTPRateFetcher to get IR info
  ctpRateFetcher rateFetcher;

  // Helper struct to pass V0 information
  struct {
    float posTrackX;
    float negTrackX;
    std::array<float, 3> pos;
    std::array<float, 3> posP;
    std::array<float, 3> negP;
    std::array<float, 3> posPosition;
    std::array<float, 3> negPosition;
    float dcaV0dau;
    float posDCAxy;
    float negDCAxy;
    float cosPA;
    float dcav0topv;
    float V0radius;
    float gammaMass;
  } v0candidate;

  o2::track::TrackParCov lPositiveTrack;
  o2::track::TrackParCov lNegativeTrack;
  o2::track::TrackParCov lPositiveTrackIU;
  o2::track::TrackParCov lNegativeTrackIU;

  ConfigurableAxis axisPt{"axisPt", {VARIABLE_WIDTH, 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.2f, 2.4f, 2.6f, 2.8f, 3.0f, 3.2f, 3.4f, 3.6f, 3.8f, 4.0f, 4.4f, 4.8f, 5.2f, 5.6f, 6.0f, 6.5f, 7.0f, 7.5f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 17.0f, 19.0f, 21.0f, 23.0f, 25.0f, 30.0f, 35.0f, 40.0f, 50.0f}, "pt axis for analysis"};
  ConfigurableAxis axisIRBinning{"axisIRBinning", {150, 0, 1500}, "Binning for the interaction rate (kHz)"};
  ConfigurableAxis axisNCounts{"axisNCounts", {100, 0, 100}, "Number of V0s"};

  void init(InitContext const&)
  {
    histos.add("GeneralQA/hInteractionRate", "hInteractionRate", kTH1F, {axisIRBinning});

    histos.add("h3dIRVsPtVsCollAssoc", "h3dIRVsPtVsCollAssoc", kTH3D, {axisIRBinning, axisPt, {2, -0.5f, 1.5f}});
    histos.get<TH3>(HIST("h3dIRVsPtVsCollAssoc"))->GetZaxis()->SetBinLabel(1, "Wrong collision");
    histos.get<TH3>(HIST("h3dIRVsPtVsCollAssoc"))->GetZaxis()->SetBinLabel(2, "Correct collision");

    histos.add("h2dIRVsNRecoV0sPerColl", "h2dIRVsNRecoV0sPerColl", kTH2D, {axisIRBinning, axisNCounts});
    histos.add("hNRecoV0s", "hNRecoV0s", kTH1D, {axisNCounts});
    histos.add("hPtParticleMC_All", "hPtParticleMC_All", kTH1D, {axisPt});
    histos.add("hPtParticleMC_CorrectAssoc", "hPtParticleMC_CorrectAssoc", kTH1D, {axisPt});

    // TPC Drift and duplicates removal
    histos.add("hTrackZ", "hTrackZ", kTH1D, {{400, -200.0f, 200.0f}});
    histos.add("hV0Z", "hV0Z", kTH1D, {{400, -200.0f, 200.0f}});

    histos.add("h2dPtVsCollAssoc_ReassocGammas", "h2dPtVsCollAssoc_ReassocGammas", kTH2D, {axisPt, {2, -0.5f, 1.5f}});
    histos.get<TH2>(HIST("h2dPtVsCollAssoc_ReassocGammas"))->GetYaxis()->SetBinLabel(1, "Wrong collision");
    histos.get<TH2>(HIST("h2dPtVsCollAssoc_ReassocGammas"))->GetYaxis()->SetBinLabel(2, "Correct collision");

    histos.add("hPtReAssocV0_CosPA", "hPtReAssocV0_CosPA", kTH1D, {axisPt});
    histos.add("hPtReAssocV0_DCADau", "hPtReAssocV0_DCADau", kTH1D, {axisPt});
    histos.add("hPtReAssocV0_CosPA_DCADau", "hPtReAssocV0_CosPA_DCADau", kTH1D, {axisPt});

    histos.add("hDCADau", "hDCADau", kTH1D, {{500, 0.0f, 5.0f}});
    histos.add("hCosPA", "hCosPA", kTH1D, {{400, 0.8f, 1.0f}});

  
    //*+-+*+-+*+-+*+-+*+-+*+-+*+-+*+-+*+-+*+-+*+-+*+-+*+-+*+-+*

    mRunNumber = 0;
    d_bz = 0;
    maxSnp = 0.85f;  // could be changed later
    maxStep = 2.00f; // could be changed later

    ccdb->setURL(ccdbConfigurations.ccdburl);
    ccdb->setCaching(true);
    ccdb->setLocalObjectValidityChecking();
    ccdb->setFatalWhenNull(false);

    if (dcaFitterConfigurations.useMatCorrType == 1) {
      LOGF(info, "TGeo correction requested, loading geometry");
      if (!o2::base::GeometryManager::isGeometryLoaded()) {
        ccdb->get<TGeoManager>(ccdbConfigurations.geoPath);
      }
    }
    if (dcaFitterConfigurations.useMatCorrType == 2) {
      LOGF(info, "LUT correction requested, will load LUT when initializing with timestamp...");
    }

    // initialize O2 2-prong fitter (only once)
    fitter.setPropagateToPCA(true);
    fitter.setMaxR(200.);
    fitter.setMinParamChange(1e-3);
    fitter.setMinRelChi2Change(0.9);
    fitter.setMaxDZIni(dcaFitterConfigurations.d_maxDZIni);
    fitter.setMaxDXYIni(dcaFitterConfigurations.d_maxDXYIni);
    fitter.setMaxChi2(1e9);
    fitter.setUseAbsDCA(dcaFitterConfigurations.d_UseAbsDCA);
    fitter.setWeightedFinalPCA(dcaFitterConfigurations.d_UseWeightedPCA);

    // Material correction in the DCA fitter
    o2::base::Propagator::MatCorrType matCorr = o2::base::Propagator::MatCorrType::USEMatCorrNONE;
    if (dcaFitterConfigurations.useMatCorrType == 1)
      matCorr = o2::base::Propagator::MatCorrType::USEMatCorrTGeo;
    if (dcaFitterConfigurations.useMatCorrType == 2)
      matCorr = o2::base::Propagator::MatCorrType::USEMatCorrLUT;
    fitter.setMatCorrType(matCorr);
  

  }

  //_______________________________________________________________________
  template <typename TMCParticle1, typename TMCParticle2, typename TMCParticles>
  int FindCommonMotherFrom2Prongs(TMCParticle1 const& p1, TMCParticle2 const& p2, const int expected_pdg1, const int expected_pdg2, const int expected_mother_pdg, TMCParticles const& mcparticles)
  {
    if (p1.globalIndex() == p2.globalIndex())
      return -1; // mc particle p1 and p2 is identical. reject.

    if (p1.pdgCode() != expected_pdg1)
      return -1;
    if (p2.pdgCode() != expected_pdg2)
      return -1;

    if (!p1.has_mothers())
      return -1;
    if (!p2.has_mothers())
      return -1;

    // LOGF(info,"original motherid1 = %d , motherid2 = %d", p1.mothersIds()[0], p2.mothersIds()[0]);

    int motherid1 = p1.mothersIds()[0];
    auto mother1 = mcparticles.iteratorAt(motherid1);
    int mother1_pdg = mother1.pdgCode();

    int motherid2 = p2.mothersIds()[0];
    auto mother2 = mcparticles.iteratorAt(motherid2);
    int mother2_pdg = mother2.pdgCode();

    // LOGF(info,"motherid1 = %d , motherid2 = %d", motherid1, motherid2);

    if (motherid1 != motherid2)
      return -1;
    if (mother1_pdg != mother2_pdg)
      return -1;
    if (mother1_pdg != expected_mother_pdg)
      return -1;
    return motherid1;
  }

  void initCCDB(aod::BCsWithTimestamps::iterator const& bc)
  {
    if (mRunNumber == bc.runNumber()) {
      return;
    }

    // In case override, don't proceed, please - no CCDB access required
    if (dcaFitterConfigurations.d_bz_input > -990) {
      d_bz = dcaFitterConfigurations.d_bz_input;
      fitter.setBz(d_bz);
      o2::parameters::GRPMagField grpmag;
      if (fabs(d_bz) > 1e-5) {
        grpmag.setL3Current(30000.f / (d_bz / 5.0f));
      }
      o2::base::Propagator::initFieldFromGRP(&grpmag);
      mVtx = ccdb->getForTimeStamp<o2::dataformats::MeanVertexObject>(ccdbConfigurations.mVtxPath, bc.timestamp());
      mRunNumber = bc.runNumber();
      return;
    }

    auto timestamp = bc.timestamp();
    o2::parameters::GRPMagField* grpmag = 0x0; 
    grpmag = ccdb->getForTimeStamp<o2::parameters::GRPMagField>(ccdbConfigurations.grpmagPath, timestamp);
    if (!grpmag) {
      LOG(fatal) << "Got nullptr from CCDB for path " << ccdbConfigurations.grpmagPath << " of object GRPMagField for timestamp " << timestamp;
    }
    o2::base::Propagator::initFieldFromGRP(grpmag);
    
    // Fetch magnetic field from ccdb for current collision
    d_bz = o2::base::Propagator::Instance()->getNominalBz();
    LOG(info) << "Retrieved GRP for timestamp " << timestamp << " with magnetic field of " << d_bz << " kG";
    mVtx = ccdb->getForTimeStamp<o2::dataformats::MeanVertexObject>(ccdbConfigurations.mVtxPath, bc.timestamp());
    mRunNumber = bc.runNumber();
    // Set magnetic field value once known
    fitter.setBz(d_bz);

    if (dcaFitterConfigurations.useMatCorrType == 2 && !lut) {
      // setMatLUT only after magfield has been initalized
      // (setMatLUT has implicit and problematic init field call if not)
      LOG(info) << "Loading material look-up table for timestamp: " << timestamp;
      lut = o2::base::MatLayerCylSet::rectifyPtrFromFile(ccdb->getForTimeStamp<o2::base::MatLayerCylSet>(ccdbConfigurations.lutPath, timestamp));
      o2::base::Propagator::Instance()->setMatLUT(lut);
    }

    mVDriftMgr.init(&ccdb->instance());
  }

  
  void updateCCDB(aod::BCsWithTimestamps::iterator const& bc)
  {
    auto timestamp = bc.timestamp();

    mVDriftMgr.update(timestamp);
  }

  float CalculateDCAStraightToPV(float X, float Y, float Z, float Px, float Py, float Pz, float pvX, float pvY, float pvZ)
  {
    return std::sqrt((std::pow((pvY - Y) * Pz - (pvZ - Z) * Py, 2) + std::pow((pvX - X) * Pz - (pvZ - Z) * Px, 2) + std::pow((pvX - X) * Py - (pvY - Y) * Px, 2)) / (Px * Px + Py * Py + Pz * Pz));
  }

  template <bool isMC, class TBCs, class TCollisions, class TTracks, typename TV0, typename TMCParticles>
  void processV0(TV0 const& v0, TMCParticles const& particlesMC, const bool isSelectedGamma)
  {
    // Get tracks
    const auto& pos = v0.template posTrack_as<TTracks>();
    const auto& ele = v0.template negTrack_as<TTracks>();
    const auto& collision = v0.template collision_as<TCollisions>(); // collision where this v0 belongs to.

    // Basic track selections
    if (pos.globalIndex() == ele.globalIndex()) {
      return;
    }
    if (!pos.hasITS() && !pos.hasTPC()) {
      return;
    }
    if (pos.hasITS() && !pos.hasTPC() && (pos.hasTRD() || pos.hasTOF())) { // remove unrealistic track. this should not happen.
      return;
    }
    if (!(!pos.hasITS() && pos.hasTPC() && !pos.hasTRD() && !pos.hasTOF())) // Is TPCOnly track? 
      return;

    if (!ele.hasITS() && !ele.hasTPC()) {
      return;
    }
    if (ele.hasITS() && !ele.hasTPC() && (ele.hasTRD() || ele.hasTOF())) { // remove unrealistic track. this should not happen.
      return;
    }
    if (!(!ele.hasITS() && ele.hasTPC() && !ele.hasTRD() && !ele.hasTOF())) // Is TPCOnly track? 
      return;

    // --------------------------------------------------------------------
    // Apply Zdrift correction
    // Calculate DCA with respect to the collision associated to the v0, not individual tracks
    gpu::gpustd::array<float, 2> dcaInfo;

    auto pTrack = getTrackParCov(pos);
    if (doVDriftMgr && !mVDriftMgr.moveTPCTrack<TBCs, TCollisions>(collision, pos, pTrack)) {
      LOGP(error, "failed correction for positive tpc track");
      return;
    }
    
    auto pTrackC = pTrack;
    pTrackC.setPID(o2::track::PID::Electron);
    o2::base::Propagator::Instance()->propagateToDCABxByBz({collision.posX(), collision.posY(), collision.posZ()}, pTrackC, 2.f, matCorr, &dcaInfo);
    auto posdcaXY = dcaInfo[0];
    auto posdcaZ = dcaInfo[1];

    auto nTrack = getTrackParCov(ele);
    if (doVDriftMgr && !mVDriftMgr.moveTPCTrack<TBCs, TCollisions>(collision, ele, nTrack)) {
      LOGP(error, "failed correction for negative tpc track");
      return;
    }
    
    auto nTrackC = nTrack;
    nTrackC.setPID(o2::track::PID::Electron);
    o2::base::Propagator::Instance()->propagateToDCABxByBz({collision.posX(), collision.posY(), collision.posZ()}, nTrackC, 2.f, matCorr, &dcaInfo);
    auto eledcaXY = dcaInfo[0];
    auto eledcaZ = dcaInfo[1];

    if (std::fabs(posdcaXY) < dcapostopv || std::fabs(eledcaXY) < dcanegtopv) {
      return;
    }

    // Initialize properly, please
    v0candidate.posDCAxy = posdcaXY;
    v0candidate.negDCAxy = eledcaXY;

    // --------------------------------------------------------------------
    // Move close to minima
    int nCand = 0;
    fitter.setCollinear(dcaFitterConfigurations.d_UseCollinearFit || v0.isCollinearV0());
    try {
      nCand = fitter.process(pTrackC, nTrackC);
    } catch (...) {      
      LOG(error) << "Exception caught in DCA fitter process call!";
      return;
    }
    if (nCand == 0) {
      return;
    }

    v0candidate.posTrackX = fitter.getTrack(0).getX();
    v0candidate.negTrackX = fitter.getTrack(1).getX();

    lPositiveTrack = fitter.getTrack(0);
    lNegativeTrack = fitter.getTrack(1);
    lPositiveTrack.getPxPyPzGlo(v0candidate.posP);
    lNegativeTrack.getPxPyPzGlo(v0candidate.negP);
    lPositiveTrack.getXYZGlo(v0candidate.posPosition);
    lNegativeTrack.getXYZGlo(v0candidate.negPosition);

    // get decay vertex coordinates
    const auto& xyz = fitter.getPCACandidate();
  
    // --------------------------------------------------------------------
    // Calculating properties

    for (int i = 0; i < 3; i++) {
      v0candidate.pos[i] = xyz[i];
    }

    v0candidate.dcaV0dau = TMath::Sqrt(fitter.getChi2AtPCACandidate());
    v0candidate.cosPA = RecoDecay::cpa(array{collision.posX(), collision.posY(), collision.posZ()}, array{v0candidate.pos[0], v0candidate.pos[1], v0candidate.pos[2]}, array{v0candidate.posP[0] + v0candidate.negP[0], v0candidate.posP[1] + v0candidate.negP[1], v0candidate.posP[2] + v0candidate.negP[2]});

    if (v0candidate.cosPA < v0cospa) {
      return;
    }

    v0candidate.dcav0topv = CalculateDCAStraightToPV(
      v0candidate.pos[0], v0candidate.pos[1], v0candidate.pos[2],
      v0candidate.posP[0] + v0candidate.negP[0],
      v0candidate.posP[1] + v0candidate.negP[1],
      v0candidate.posP[2] + v0candidate.negP[2],
      collision.posX(), collision.posY(), collision.posZ());

    v0candidate.V0radius = RecoDecay::sqrtSumOfSquares(v0candidate.pos[0], v0candidate.pos[1]);
    if (v0candidate.V0radius < v0radius) {
      return;
    }

    // if (v0candidate.V0radius > maxX + margin_r_tpc) {
    //   return;
    // }

    auto px = v0candidate.posP[0] + v0candidate.negP[0];
    auto py = v0candidate.posP[1] + v0candidate.negP[1];
    auto pz = v0candidate.posP[2] + v0candidate.negP[2];
    auto lPt = RecoDecay::sqrtSumOfSquares(v0candidate.posP[0] + v0candidate.negP[0], v0candidate.posP[1] + v0candidate.negP[1]);
    auto lPtotal = RecoDecay::sqrtSumOfSquares(lPt, v0candidate.posP[2] + v0candidate.negP[2]);
    auto lLengthTraveled = RecoDecay::sqrtSumOfSquares(v0candidate.pos[0] - collision.posX(), v0candidate.pos[1] - collision.posY(), v0candidate.pos[2] - collision.posZ());


    // Momentum range check
    if (lPt < minimumPt || lPt > maximumPt) {
      return; // reject if not within desired window
    }

    // Daughter eta check
    if (TMath::Abs(RecoDecay::eta(std::array{v0candidate.posP[0], v0candidate.posP[1], v0candidate.posP[2]})) > maxDaughterEta ||
        TMath::Abs(RecoDecay::eta(std::array{v0candidate.negP[0], v0candidate.negP[1], v0candidate.negP[2]})) > maxDaughterEta) {
      return; // reject - daughters have too large eta to be reliable for MC corrections
    }

    v0candidate.gammaMass = RecoDecay::m(array{array{v0candidate.posP[0], v0candidate.posP[1], v0candidate.posP[2]}, array{v0candidate.negP[0], v0candidate.negP[1], v0candidate.negP[2]}}, array{o2::constants::physics::MassElectron, o2::constants::physics::MassElectron});

    pca_map[std::make_tuple(v0.globalIndex(), collision.globalIndex(), pos.globalIndex(), ele.globalIndex())] = v0candidate.dcaV0dau;
    cospa_map[std::make_tuple(v0.globalIndex(), collision.globalIndex(), pos.globalIndex(), ele.globalIndex())] = v0candidate.cosPA;

    if (!ele.has_mcParticle() || !pos.has_mcParticle()) {
      return;
    }

    auto lMCNegTrack = ele.template mcParticle_as<aod::McParticles>();
    auto lMCPosTrack = pos.template mcParticle_as<aod::McParticles>();  
    float mcpt = -1.0;
          
    int v0id = FindCommonMotherFrom2Prongs(lMCPosTrack, lMCNegTrack, PDGCodePosDau, PDGCodeNegDau, PDGCodeMother, particlesMC);

    if (v0id <= 0) {
      return;
    }

    auto mcv0 = particlesMC.iteratorAt(v0id);

    // TODO: Add histos here
    int correctMcCollisionIndex = mcv0.mcCollisionId();
    mcpt = mcv0.pt();

    bool collisionAssociationOK = false;
    if (correctMcCollisionIndex > -1 && correctMcCollisionIndex == collision.mcCollisionId()) {
      collisionAssociationOK = true;
    }

    mcpT_map[std::make_tuple(v0.globalIndex(), collision.globalIndex(), pos.globalIndex(), ele.globalIndex())] = mcpt;
    collassoc_map[std::make_tuple(v0.globalIndex(), collision.globalIndex(), pos.globalIndex(), ele.globalIndex())] = collisionAssociationOK;
    if (isSelectedGamma){
      histos.fill(HIST("h2dPtVsCollAssoc_ReassocGammas"), mcpt, collisionAssociationOK); 
      histos.fill(HIST("hDCADau"), v0candidate.dcaV0dau); 
      histos.fill(HIST("hCosPA"), v0candidate.cosPA); 
    }
    else{
      histos.fill(HIST("hTrackZ"), pTrack.getZ());
      histos.fill(HIST("hTrackZ"), nTrack.getZ());
      histos.fill(HIST("hV0Z"), v0candidate.posP[2]);
    }
  }

  std::map<std::tuple<int64_t, int64_t, int64_t, int64_t>, float> pca_map;      // (v0.globalIndex(), collision.globalIndex(), pos.globalIndex(), ele.globalIndex()) -> pca
  std::map<std::tuple<int64_t, int64_t, int64_t, int64_t>, float> cospa_map;    // (v0.globalIndex(), collision.globalIndex(), pos.globalIndex(), ele.globalIndex()) -> cospa
  std::map<std::tuple<int64_t, bool, int64_t, int64_t>, float> mcpT_map;        // (v0.globalIndex(), collision.mcIndex(), pos.globalIndex(), ele.globalIndex()) -> cospa
  std::map<std::tuple<int64_t, bool, int64_t, int64_t>, bool> collassoc_map;   // (v0.globalIndex(), collision.mcIndex(), pos.globalIndex(), ele.globalIndex()) -> cospa
  std::vector<std::pair<int64_t, int64_t>> stored_v0Ids;                        // (pos.globalIndex(), ele.globalIndex())
  std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> stored_fullv0Ids; // (v0.globalIndex(), collision.globalIndex(), pos.globalIndex(), ele.globalIndex())
  std::unordered_map<int64_t, int> nv0_map;                                     // map collisionId -> nv0

  template <bool isMC, bool enableFilter, typename TCollisions, typename TV0s, typename TTracks, typename TBCs, typename TMCParticles>
  void build(TCollisions const& collisions, TV0s const& v0s, TTracks const&, TBCs const&, TMCParticles const& mcparticles)
  {
    for (const auto& collision : collisions) {
      if constexpr (isMC) {
        if (!collision.has_mcCollision()) {
          continue;
        }
      }

      // TODO: include event selection here? 

      nv0_map[collision.globalIndex()] = 0;

      //const auto& bc = collision.template foundBC_as<aod::BCsWithTimestamps>();
      auto bc = collision.template foundBC_as<aod::BCsWithTimestamps>();
      
      initCCDB(bc);
      updateCCDB(bc); // delay update until is needed

      const auto& v0s_per_coll = v0s.sliceBy(perCollision, collision.globalIndex());
      for (const auto& v0 : v0s_per_coll) {
        processV0<isMC, TBCs, TCollisions, TTracks>(v0, mcparticles, false);
      } // end of v0 loop
    } // end of collision loop

    stored_v0Ids.reserve(pca_map.size());     // number of photon candidates per DF
    stored_fullv0Ids.reserve(pca_map.size()); // number of photon candidates per DF

    // find minimal pca
    for (const auto& [key, value] : pca_map) {
      auto v0Id = std::get<0>(key);
      auto collisionId = std::get<1>(key);
      auto posId = std::get<2>(key);
      auto eleId = std::get<3>(key);
      float v0pca = value;
      float cospa = cospa_map[key];
      bool is_closest_v0 = true;
      bool is_most_aligned_v0 = true;
      float v0mcpT = mcpT_map[key];
      bool isCorrectColl = collassoc_map[key];

      for (const auto& [key_tmp, value_tmp] : pca_map) {
        auto v0Id_tmp = std::get<0>(key_tmp);
        auto collisionId_tmp = std::get<1>(key_tmp);
        auto posId_tmp = std::get<2>(key_tmp);
        auto eleId_tmp = std::get<3>(key_tmp);
        float v0pca_tmp = value_tmp;
        float cospa_tmp = cospa_map[key_tmp];

        if (v0Id == v0Id_tmp) { // skip exactly the same v0
          continue;
        }

        if (collisionId != collisionId_tmp && eleId == eleId_tmp && posId == posId_tmp && cospa < cospa_tmp) { // same ele and pos, but attached to different collision
          // LOGF(info, "!reject! | collision id = %d | posid1 = %d , eleid1 = %d , posid2 = %d , eleid2 = %d , cospa1 = %f , cospa2 = %f", collisionId, posId, eleId, posId_tmp, eleId_tmp, cospa, cospa_tmp);
          is_most_aligned_v0 = false;
          break;
        }

        if ((eleId == eleId_tmp || posId == posId_tmp) && v0pca > v0pca_tmp) {
          // LOGF(info, "!reject! | collision id = %d | posid1 = %d , eleid1 = %d , posid2 = %d , eleid2 = %d , pca1 = %f , pca2 = %f", collisionId, posId, eleId, posId_tmp, eleId_tmp, v0pca, v0pca_tmp);
          is_closest_v0 = false;
          break;
        }
      } // end of pca_map tmp loop

      bool is_stored = std::find(stored_v0Ids.begin(), stored_v0Ids.end(), std::make_pair(posId, eleId)) != stored_v0Ids.end();

      if (is_most_aligned_v0 && isCorrectColl && !is_stored) histos.fill(HIST("hPtReAssocV0_CosPA"), v0mcpT); 
      if (is_closest_v0 && isCorrectColl && !is_stored) histos.fill(HIST("hPtReAssocV0_DCADau"), v0mcpT); 
      if (is_most_aligned_v0 && is_closest_v0 && isCorrectColl && !is_stored) histos.fill(HIST("hPtReAssocV0_CosPA_DCADau"), v0mcpT); 

      if (is_closest_v0 && is_most_aligned_v0 && !is_stored) {

        stored_v0Ids.emplace_back(std::make_pair(posId, eleId));
        stored_fullv0Ids.emplace_back(std::make_tuple(v0Id, collisionId, posId, eleId));
        nv0_map[collisionId]++;          
      }
    } // end of pca_map loop
    
    for (auto& fullv0Id : stored_fullv0Ids) {
      auto v0Id = std::get<0>(fullv0Id);    
      auto v0 = v0s.rawIteratorAt(v0Id);
  
      // Save the selected v0!
      processV0<isMC, TBCs, TCollisions, TTracks>(v0, mcparticles, true);
    } // end of fullv0Id loop

    pca_map.clear();
    cospa_map.clear();
    mcpT_map.clear();
    collassoc_map.clear();
    nv0_map.clear();
    stored_v0Ids.clear();
    stored_v0Ids.shrink_to_fit();
    stored_fullv0Ids.clear();
    stored_fullv0Ids.shrink_to_fit();
  } // end of build

  void processBuildMCAssociated(soa::Join<aod::Collisions, aod::EvSels, aod::McCollisionLabels> const& collisions, aod::V0s const& v0table, LabeledTracksExtra const&, aod::McParticles const& particlesMC, aod::BCsWithTimestamps const&)
  {
    std::unordered_map<int, int> mcV0Counts;
    std::unordered_set<int> filledMcPt; // Set to keep track of filled particleMC ids
    for (auto& collision : collisions) {
      std::unordered_map<int, int> mcV0CountsPerColl;
      auto V0s = v0table.sliceBy(perCollision, collision.globalIndex());

      double interactionRate = 500;
      auto bc = collision.template foundBC_as<aod::BCsWithTimestamps>();
      if (fGetIR) interactionRate = rateFetcher.fetch(ccdb.service, bc.timestamp(), bc.runNumber(), irSource) * 1.e-3;
      histos.fill(HIST("GeneralQA/hInteractionRate"), interactionRate);

      for (auto const& v0 : V0s) {

        auto lNegTrack = v0.template negTrack_as<LabeledTracksExtra>();
        auto lPosTrack = v0.template posTrack_as<LabeledTracksExtra>();
        float mcpt = -1.0;

        if (!lNegTrack.has_mcParticle() || !lPosTrack.has_mcParticle()) {
          continue;
        }

        auto lMCNegTrack = lNegTrack.template mcParticle_as<aod::McParticles>();
        auto lMCPosTrack = lPosTrack.template mcParticle_as<aod::McParticles>();  
              
        int v0id = FindCommonMotherFrom2Prongs(lMCPosTrack, lMCNegTrack, PDGCodePosDau, PDGCodeNegDau, PDGCodeMother, particlesMC);

        if (v0id <= 0) {
          continue;
        }

        auto mcv0 = particlesMC.iteratorAt(v0id);
        mcV0Counts[v0id]++;
        mcV0CountsPerColl[v0id]++;
        
        int correctMcCollisionIndex = mcv0.mcCollisionId();
        mcpt = mcv0.pt();

        bool collisionAssociationOK = false;
        if (correctMcCollisionIndex > -1 && correctMcCollisionIndex == collision.mcCollisionId()) {
          collisionAssociationOK = true;
          histos.fill(HIST("hPtParticleMC_CorrectAssoc"), mcpt);
        }

        // Fill only if this particleMC has not been filled before
        if (filledMcPt.insert(v0id).second) {
          histos.fill(HIST("hPtParticleMC_All"), mcpt);          
        }

        histos.fill(HIST("h3dIRVsPtVsCollAssoc"), interactionRate, mcpt, collisionAssociationOK); 
      }

      for (const auto& [v0id, count] : mcV0CountsPerColl) {
        histos.fill(HIST("h2dIRVsNRecoV0sPerColl"), interactionRate, count);
      }
    }
    for (const auto& [v0id, count] : mcV0Counts) {
      histos.fill(HIST("hNRecoV0s"), count);
    }
  }

  void processMCBuildPhoton(soa::Join<aod::Collisions, aod::EvSels, aod::McCollisionLabels> const& collisions, aod::V0s const& v0table, LabeledTracksExtra const& tracks, aod::BCsWithTimestamps const& bcs, aod::McParticles const& particlesMC)
  {
    build<true, false>(collisions, v0table, tracks, bcs, particlesMC);
  }

  PROCESS_SWITCH(v0assoqa, processBuildMCAssociated, "Check wrong v0-to-collision association", true);
  PROCESS_SWITCH(v0assoqa, processMCBuildPhoton, "Try to reproduce the building process of photonconversionbuilder.cxx", false);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{adaptAnalysisTask<v0assoqa>(cfgc)};
}