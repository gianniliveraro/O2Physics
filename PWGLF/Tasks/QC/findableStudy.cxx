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
// Strangeness findable study
//
// --- deals with derived data that has been specifically
//     generated to do the findable exercise.
//
//    Comments, questions, complaints, suggestions?
//    Please write to:
//    david.dobrigkeit.chinellato@cern.ch
//

#include <Math/Vector4D.h>
#include <cmath>
#include <array>
#include <cstdlib>

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "ReconstructionDataFormats/Track.h"
#include "Common/Core/RecoDecay.h"
#include "Common/Core/trackUtilities.h"
#include "PWGLF/DataModel/LFStrangenessTables.h"
#include "PWGLF/DataModel/LFStrangenessPIDTables.h"
#include "PWGLF/DataModel/LFParticleIdentification.h"
#include "Common/Core/TrackSelection.h"
#include "Common/DataModel/McCollisionExtra.h"
#include "Common/DataModel/TrackSelectionTables.h"
#include "Common/DataModel/EventSelection.h"
#include "Common/DataModel/Centrality.h"
#include "Common/DataModel/PIDResponse.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GeometryManager.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "CCDB/BasicCCDBManager.h"
#include "PWGLF/Utils/v0SelectionBits.h"
#include "PWGLF/Utils/v0SelectionGroup.h"
#include "PWGLF/Utils/v0SelectionTools.h"

#include <TFile.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TLorentzVector.h>
#include <TPDGCode.h>
#include <TDatabasePDG.h>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;

using recoStraCollisions = soa::Join<aod::StraCollisions, aod::StraEvSels, aod::StraCents, aod::StraRawCents, aod::StraCollLabels>;
using reconstructedV0s = soa::Join<aod::V0CoreMCLabels, aod::V0Cores, aod::V0FoundTags, aod::V0MCCollRefs, aod::V0CollRefs, aod::V0Extras, aod::V0TOFPIDs, aod::V0TOFNSigmas>;
using reconstructedV0sNoMC = soa::Join<aod::V0Cores, aod::V0Extras>;

using dauTracks = soa::Join<aod::DauTrackExtras, aod::DauTrackTPCPIDs>;

// simple checkers, but ensure 64 bit integers
#define bitset(var, nbit) ((var) |= (static_cast<uint64_t>(1) << static_cast<uint64_t>(nbit)))
#define bitcheck(var, nbit) ((var) & (static_cast<uint64_t>(1) << static_cast<uint64_t>(nbit)))

struct findableStudy {
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  // master PDG code selection
  Configurable<int> pdgCode{"pdgCode", 310, "PDG code to select"};
  Configurable<int> pdgCodeMother{"pdgCodeMother", -1, "PDG code mother to select"};
  Configurable<bool> skipITSonly{"skipITSonly", true, "skip reco V0s if an ITS-only (no TPC) prong present"};
  Configurable<bool> selStandardV0{"selStandardV0", true, "skip anything other than a standard V0 (V0Type==1)"};

  ConfigurableAxis axisPt{"axisPt", {VARIABLE_WIDTH, 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.2f, 2.4f, 2.6f, 2.8f, 3.0f, 3.2f, 3.4f, 3.6f, 3.8f, 4.0f, 4.4f, 4.8f, 5.2f, 5.6f, 6.0f, 6.5f, 7.0f, 7.5f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 17.0f, 19.0f, 21.0f, 23.0f, 25.0f, 30.0f, 35.0f, 40.0f, 50.0f}, "pt axis for analysis"};
  ConfigurableAxis axisCentrality{"axisCentrality", {VARIABLE_WIDTH, 0.0f, 5.0f, 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f, 90.0f}, "Centrality"};
  ConfigurableAxis axisV0Radius{"axisV0Radius", {180, 0.0f, 180.0f}, "V0 2D radius (cm)"};
  ConfigurableAxis axisDCAdau{"axisDCAdau", {30, 0.0f, 3.0f}, "DCA (cm)"};
  ConfigurableAxis axisRadiusResolution{"axisRadiusResolution", {100, 0.0f, 10.0f}, "Radius Resolution"};

  // +-~-+-~-+-~-+-~-+-~-+-~-+-~-+-~-+-~-+-~-+-~-+-~-+-~-+-~-+-~-+
  // Full wrapper for configurables related to actual analysis
  Configurable<v0SelectionGroup> v0Selections{"v0Selections", {}, "V0 selection criteria for analysis"};
  // +-~-+-~-+-~-+-~-+-~-+-~-+-~-+-~-+-~-+-~-+-~-+-~-+-~-+-~-+-~-+

  // pack track quality but separte also afterburner
  // dynamic range: 0-31
  enum selection : int { hasTPC = 0,
                         hasITSTracker,
                         hasITSAfterburner,
                         hasTRD,
                         hasTOF };

  uint64_t maskTopological;
  uint64_t maskTrackProperties;

  uint64_t maskK0ShortSpecific;
  uint64_t maskLambdaSpecific;
  uint64_t maskAntiLambdaSpecific;

  uint64_t maskSelectionK0Short;
  uint64_t maskSelectionLambda;
  uint64_t maskSelectionAntiLambda;

  void init(InitContext const&)
  {
    v0Selections->PrintSelections(); // for the logs

    v0Selections->provideMasks(maskTopological, maskTrackProperties, maskK0ShortSpecific, maskLambdaSpecific, maskAntiLambdaSpecific);

    // Primary particle selection, central to analysis
    maskSelectionK0Short = maskTopological | maskTrackProperties | maskK0ShortSpecific;
    maskSelectionLambda = maskTopological | maskTrackProperties | maskLambdaSpecific;
    maskSelectionAntiLambda = maskTopological | maskTrackProperties | maskAntiLambdaSpecific;

    // Event counting
    histos.add("hCentrality", "hCentrality", kTH1D, {axisCentrality});

    // Duplicate counting
    histos.add("hNRecoV0s", "hNRecoV0s", kTH1D, {{50, -0.5, 49.5f}});
    histos.add("hNRecoV0sWithTPC", "hNRecoV0sWithTPC", kTH1D, {{50, -0.5, 49.5f}});

    histos.add("h3dPtVsCentralityNRecoV0s_Findable",  "h3dPtVsCentralityNRecoV0s_Findable", kTH3D, {axisCentrality, axisPt, {50, -0.5, 49.5f}});
    histos.add("h3dPtVsCentralityNRecoV0s_AcceptablyTracked_TPCOther",  "h3dPtVsCentralityNRecoV0s_AcceptablyTracked_TPCOther", kTH3D, {axisCentrality, axisPt, {50, -0.5, 49.5f}});
    histos.add("h3dPtVsCentralityNRecoV0s_AcceptablyTracked_anyTPCOnly", "h3dPtVsCentralityNRecoV0s_AcceptablyTracked_anyTPCOnly", kTH3D, {axisCentrality, axisPt, {50, -0.5, 49.5f}});
    histos.add("h3dPtVsCentralityNRecoV0s_AcceptablyTracked_TPCOnly", "h3dPtVsCentralityNRecoV0s_AcceptablyTracked_TPCOnly", kTH3D, {axisCentrality, axisPt, {50, -0.5, 49.5f}});
    histos.add("h3dPtVsCentralityNRecoV0s_Found_TPCOther", "h3dPtVsCentralityNRecoV0s_Found_TPCOther", kTH3D, {axisCentrality, axisPt, {50, -0.5, 49.5f}});
    histos.add("h3dPtVsCentralityNRecoV0s_Found_anyTPCOnly", "h3dPtVsCentralityNRecoV0s_Found_anyTPCOnly", kTH3D, {axisCentrality, axisPt, {50, -0.5, 49.5f}});
    histos.add("h3dPtVsCentralityNRecoV0s_Found_TPCOnly", "h3dPtVsCentralityNRecoV0s_Found_TPCOnly", kTH3D, {axisCentrality, axisPt, {50, -0.5, 49.5f}});

    // With wrong collision
    histos.add("hNRecoV0sWrongColl", "hNRecoV0sWrongColl", kTH1D, {{50, -0.5, 49.5f}});
    histos.add("h3dPtVsCentralityNRecoV0sWrongColl_AcceptablyTracked_TPCOther",  "h3dPtVsCentralityNRecoV0sWrongColl_AcceptablyTracked_TPCOther", kTH3D, {axisCentrality, axisPt, {50, -0.5, 49.5f}});
    histos.add("h3dPtVsCentralityNRecoV0sWrongColl_AcceptablyTracked_anyTPCOnly", "h3dPtVsCentralityNRecoV0sWrongColl_AcceptablyTracked_anyTPCOnly", kTH3D, {axisCentrality, axisPt, {50, -0.5, 49.5f}});
    histos.add("h3dPtVsCentralityNRecoV0sWrongColl_AcceptablyTracked_TPCOnly", "h3dPtVsCentralityNRecoV0sWrongColl_AcceptablyTracked_TPCOnly", kTH3D, {axisCentrality, axisPt, {50, -0.5, 49.5f}});
    histos.add("h3dPtVsCentralityNRecoV0sWrongColl_Found_TPCOther", "h3dPtVsCentralityNRecoV0sWrongColl_Found_TPCOther", kTH3D, {axisCentrality, axisPt, {50, -0.5, 49.5f}});
    histos.add("h3dPtVsCentralityNRecoV0sWrongColl_Found_anyTPCOnly", "h3dPtVsCentralityNRecoV0sWrongColl_Found_anyTPCOnly", kTH3D, {axisCentrality, axisPt, {50, -0.5, 49.5f}});
    histos.add("h3dPtVsCentralityNRecoV0sWrongColl_Found_TPCOnly", "h3dPtVsCentralityNRecoV0sWrongColl_Found_TPCOnly", kTH3D, {axisCentrality, axisPt, {50, -0.5, 49.5f}});

    // For broad correctness check
    histos.add("hFoundVsTracksOK", "hFoundVsTracksOK", kTH2D, {{2, -0.5, 1.5f}, {2, -0.5, 1.5f}});

    // Global findable
    histos.add("h2dPtVsCentrality_Findable", "hPtVsCentrality_Findable", kTH2D, {axisCentrality, axisPt});

    // Acceptably (for svertexer) tracked
    histos.add("h2dPtVsCentrality_AcceptablyTracked", "h2dPtVsCentrality_AcceptablyTracked", kTH2D, {axisCentrality, axisPt});
    histos.add("h2dPtVsCentrality_AcceptTracked_ITSOnly", "h3dPtVsCentrality_AcceptTracked_ITSOnly", kTH2D, {axisCentrality, axisPt});
    histos.add("h2dPtVsCentrality_AcceptTracked_TPCOnly", "h3dPtVsCentrality_AcceptTracked_TPCOnly", kTH2D, {axisCentrality, axisPt});
    histos.add("h2dPtVsCentrality_AcceptTracked_ITSTPC", "h3dPtVsCentrality_AcceptTracked_ITSTPC", kTH2D, {axisCentrality, axisPt});
    histos.add("h2dPtVsCentrality_AcceptTracked_ITSTPCOuter", "h3dPtVsCentrality_AcceptTracked_ITSTPCOuter", kTH2D, {axisCentrality, axisPt});

    // V0Type check
    histos.add("hv0Type", "hv0Type", kTH1D, {{6, -0.5, 5.5f}});

    // Topological analysis of Acceptably tracked
    // Radius
    histos.add("h2dPtVsCentrality_Findable_Radius", "h2dPtVsCentrality_Findable_Radius", kTH3F, {axisCentrality, axisPt, axisV0Radius});
    histos.add("h3dPtVsCentralityRadius_AcceptTracked_ITSOnly", "h3dPtVsCentralityRadius_AcceptTracked_ITSOnly", kTH3F, {axisCentrality, axisPt, axisV0Radius});
    histos.add("h3dPtVsCentralityRadius_AcceptTracked_TPCOnly", "h3dPtVsCentralityRadius_AcceptTracked_TPCOnly", kTH3F, {axisCentrality, axisPt, axisV0Radius});
    histos.add("h3dPtVsCentralityRadius_AcceptTracked_ITSTPC", "h3dPtVsCentralityRadius_AcceptTracked_ITSTPC", kTH3F, {axisCentrality, axisPt, axisV0Radius});
    histos.add("h3dPtVsCentralityRadius_AcceptTracked_ITSTPCOuter", "h3dPtVsCentralityRadius_AcceptTracked_ITSTPCOuter", kTH3F, {axisCentrality, axisPt, axisV0Radius});
    
    histos.add("h3dPtVsCentralityRadius_Found_ITSOnly", "h3dPtVsCentralityRadius_Found_ITSOnly", kTH3F, {axisCentrality, axisPt, axisV0Radius});
    histos.add("h3dPtVsCentralityRadius_Found_TPCOnly", "h3dPtVsCentralityRadius_Found_TPCOnly", kTH3F, {axisCentrality, axisPt, axisV0Radius});
    histos.add("h3dPtVsCentralityRadius_Found_ITSTPC", "h3dPtVsCentralityRadius_Found_ITSTPC", kTH3F, {axisCentrality, axisPt, axisV0Radius});
    histos.add("h3dPtVsCentralityRadius_Found_ITSTPCOuter", "h3dPtVsCentralityRadius_Found_ITSTPCOuter", kTH3F, {axisCentrality, axisPt, axisV0Radius});
    
    // Radius resolution
    histos.add("hRadiusResolution_AcceptTracked_ITSOnly","hRadiusResolution_AcceptTracked_ITSOnly" , kTH1F, {axisRadiusResolution});
    histos.add("hRadiusResolution_AcceptTracked_TPCOnly","hRadiusResolution_AcceptTracked_TPCOnly" , kTH1F, {axisRadiusResolution});
    histos.add("hRadiusResolution_AcceptTracked_ITSTPC", "hRadiusResolution_AcceptTracked_ITSTPC", kTH1F, {axisRadiusResolution});
    histos.add("hRadiusResolution_AcceptTracked_ITSTPCOuter", "hRadiusResolution_AcceptTracked_ITSTPCOuter", kTH1F, {axisRadiusResolution});
    
    histos.add("hRadiusResolution_Found_ITSOnly","hRadiusResolution_Found_ITSOnly" , kTH1F, {axisRadiusResolution});
    histos.add("hRadiusResolution_Found_TPCOnly","hRadiusResolution_Found_TPCOnly" , kTH1F, {axisRadiusResolution});
    histos.add("hRadiusResolution_Found_ITSTPC", "hRadiusResolution_Found_ITSTPC", kTH1F, {axisRadiusResolution});
    histos.add("hRadiusResolution_Found_ITSTPCOuter", "hRadiusResolution_Found_ITSTPCOuter", kTH1F, {axisRadiusResolution});
    
    // DCADau
    histos.add("h3dPtVsCentralityDCADau_AcceptTracked_ITSOnly", "h3dPtVsCentralityDCADau_AcceptTracked_ITSOnly", kTH3F, {axisCentrality, axisPt, axisDCAdau});
    histos.add("h3dPtVsCentralityDCADau_AcceptTracked_TPCOnly", "h3dPtVsCentralityDCADau_AcceptTracked_TPCOnly", kTH3F, {axisCentrality, axisPt, axisDCAdau});
    histos.add("h3dPtVsCentralityDCADau_AcceptTracked_ITSTPC", "h3dPtVsCentralityDCADau_AcceptTracked_ITSTPC", kTH3F, {axisCentrality, axisPt, axisDCAdau});
    histos.add("h3dPtVsCentralityDCADau_AcceptTracked_ITSTPCOuter", "h3dPtVsCentralityDCADau_AcceptTracked_ITSTPCOuter", kTH3F, {axisCentrality, axisPt, axisDCAdau});
    
    histos.add("h3dPtVsCentralityDCADau_Found_ITSOnly", "h3dPtVsCentralityDCADau_Found_ITSOnly", kTH3F, {axisCentrality, axisPt, axisDCAdau});
    histos.add("h3dPtVsCentralityDCADau_Found_TPCOnly", "h3dPtVsCentralityDCADau_Found_TPCOnly", kTH3F, {axisCentrality, axisPt, axisDCAdau});
    histos.add("h3dPtVsCentralityDCADau_Found_ITSTPC", "h3dPtVsCentralityDCADau_Found_ITSTPC", kTH3F, {axisCentrality, axisPt, axisDCAdau});
    histos.add("h3dPtVsCentralityDCADau_Found_ITSTPCOuter", "h3dPtVsCentralityDCADau_Found_ITSTPCOuter", kTH3F, {axisCentrality, axisPt, axisDCAdau});
    
    // Found in any capacity, including ITSonly
    histos.add("h2dPtVsCentrality_FoundAny", "h2dPtVsCentrality_FoundAny", kTH2D, {axisCentrality, axisPt});

    // Found with prongs with the TPC (typical analysis setting)
    histos.add("h2dPtVsCentrality_Found", "h2dPtVsCentrality_Found", kTH2D, {axisCentrality, axisPt});
    histos.add("h2dPtVsCentrality_Found_ITSOnly", "h2dPtVsCentrality_Found_ITSOnly", kTH2D, {axisCentrality, axisPt});
    histos.add("h2dPtVsCentrality_Found_TPCOnly", "h2dPtVsCentrality_Found_TPCOnly", kTH2D, {axisCentrality, axisPt});
    histos.add("h2dPtVsCentrality_Found_ITSTPC", "h2dPtVsCentrality_Found_ITSTPC", kTH2D, {axisCentrality, axisPt});
    histos.add("h2dPtVsCentrality_Found_ITSTPCOuter", "h2dPtVsCentrality_Found_ITSTPCOuter", kTH2D, {axisCentrality, axisPt});
    
    // Found in loop (may have duplicates, meant as cross-check too)
    histos.add("h2dPtVsCentrality_FoundInLoop", "h2dPtVsCentrality_FoundInLoop", kTH2D, {axisCentrality, axisPt});

    // Passes analysis-level track quality checks
    histos.add("h2dPtVsCentrality_PassesTrackQuality", "h2dPtVsCentrality_Analysis_PassesTrackQuality", kTH2D, {axisCentrality, axisPt});

    // Passes analysis-level topological selection criteria
    histos.add("h2dPtVsCentrality_PassesTopological", "h2dPtVsCentrality_PassesTopological", kTH2D, {axisCentrality, axisPt});

    // Passes analysis-level species-specific
    histos.add("h2dPtVsCentrality_PassesThisSpecies", "h2dPtVsCentrality_PassesThisSpecies", kTH2D, {axisCentrality, axisPt});

    // one-on-one test for topology, operating with track quality and species-specific checks ON
    histos.add("h2dPtVsCentrality_V0Radius", "h2dPtVsCentrality_V0Radius", kTH2D, {axisCentrality, axisPt});
    histos.add("h2dPtVsCentrality_V0RadiusMax", "h2dPtVsCentrality_V0RadiusMax", kTH2D, {axisCentrality, axisPt});
    histos.add("h2dPtVsCentrality_V0CosPA", "h2dPtVsCentrality_V0CosPA", kTH2D, {axisCentrality, axisPt});
    histos.add("h2dPtVsCentrality_DcaPosToPV", "h2dPtVsCentrality_DcaPosToPV", kTH2D, {axisCentrality, axisPt});
    histos.add("h2dPtVsCentrality_DcaNegToPV", "h2dPtVsCentrality_DcaNegToPV", kTH2D, {axisCentrality, axisPt});
    histos.add("h2dPtVsCentrality_DcaV0Dau", "h2dPtVsCentrality_DcaV0Dau", kTH2D, {axisCentrality, axisPt});

    // Track quality tests in steps
    histos.add("h2dTrackPropAcceptablyTracked", "h2dTrackPropAcceptablyTracked", kTH3D, {{32, -0.5, 31.5f}, {32, -0.5, 31.5f}, axisCentrality});
    histos.add("h2dTrackPropFound", "h2dTrackPropFound", kTH3D, {{32, -0.5, 31.5f}, {32, -0.5, 31.5f}, axisCentrality});
    histos.add("h2dTrackPropAnalysisTracks", "h2dTrackPropAnalysisTracks", kTH3D, {{32, -0.5, 31.5f}, {32, -0.5, 31.5f}, axisCentrality});
    histos.add("h2dTrackPropAnalysisTopo", "h2dTrackPropAnalysisTopo", kTH3D, {{32, -0.5, 31.5f}, {32, -0.5, 31.5f}, axisCentrality});
    histos.add("h2dTrackPropAnalysisSpecies", "h2dTrackPropAnalysisSpecies", kTH3D, {{32, -0.5, 31.5f}, {32, -0.5, 31.5f}, axisCentrality});
  }

  void processEvents(
    recoStraCollisions::iterator const& collision // reco collisions for collision counting
  )
  {
    histos.fill(HIST("hCentrality"), collision.centFT0C());
  }

  void processDebugCrossCheck(
    reconstructedV0sNoMC::iterator const& recV0, // reco V0s for cross-check
    dauTracks const&                             // daughter track extras
  )
  {
    if (recV0.v0Type() == 1) {
      // de-reference daughter track extras
      auto pTrack = recV0.posTrackExtra_as<dauTracks>();
      auto nTrack = recV0.negTrackExtra_as<dauTracks>();

      // cross-check correctness of new getter
      if (pTrack.hasTPC() && !pTrack.hasITS() && !pTrack.hasTRD() && !pTrack.hasTOF()) {
        LOGF(info, "X-check: Positive track is TPC only and this is a found V0. Puzzling!");
      }
      if (nTrack.hasTPC() && !nTrack.hasITS() && !nTrack.hasTRD() && !nTrack.hasTOF()) {
        LOGF(info, "X-check: Negative track is TPC only and this is a found V0. Puzzling!");
      }
    }
  }

  void processDebugCrossCheckMC(
    reconstructedV0s::iterator const& recV0, // reco V0s for cross-check
    dauTracks const&                         // daughter track extras
  )
  {
    if (recV0.isFound() && recV0.v0Type() == 1) {
      // de-reference daughter track extras
      auto pTrack = recV0.posTrackExtra_as<dauTracks>();
      auto nTrack = recV0.negTrackExtra_as<dauTracks>();

      // cross-check correctness of new getter
      if (pTrack.hasTPC() && !pTrack.hasITS() && !pTrack.hasTRD() && !pTrack.hasTOF()) {
        LOGF(info, "X-check: Positive track is TPC only and this is a found V0. Puzzling!");
      }
      if (nTrack.hasTPC() && !nTrack.hasITS() && !nTrack.hasTRD() && !nTrack.hasTOF()) {
        LOGF(info, "X-check: Negative track is TPC only and this is a found V0. Puzzling!");
      }
    }
  }

  void processV0s(
    aod::V0MCCores::iterator const& v0,               // non-duplicated MC V0 decays
    soa::SmallGroups<reconstructedV0s> const& recv0s, // reconstructed versions of the v0
    recoStraCollisions const&,                        // reco collisions for de-reference
    aod::StraMCCollisions const&,                     // MC collisions for de-reference
    dauTracks const&                                  // daughter track extras
  )
  {
    int pdgCodePositive = 211;
    int pdgCodeNegative = -211;
    if (pdgCode == 3122)
      pdgCodePositive = 2212;
    if (pdgCode == -3122)
      pdgCodePositive = -2212;
    if (pdgCode == 22) {
      pdgCodePositive = -11;
      pdgCodeNegative = +11;
    }

    if (pdgCodeMother!=-1 && v0.pdgCodeMother()!=pdgCodeMother){
      return;
    }
    
    if (v0.pdgCode() != pdgCode || v0.pdgCodePositive() != pdgCodePositive || v0.pdgCodeNegative() != pdgCodeNegative)
      return;
    if (!v0.isPhysicalPrimary())
      return;

    float rapidity = 2.0;
    if (pdgCode == 310)
      rapidity = RecoDecay::y(std::array{v0.pxPosMC() + v0.pxNegMC(), v0.pyPosMC() + v0.pyNegMC(), v0.pzPosMC() + v0.pzNegMC()}, o2::constants::physics::MassKaonNeutral);
    if (pdgCode == 22)
      rapidity = RecoDecay::y(std::array{v0.pxPosMC() + v0.pxNegMC(), v0.pyPosMC() + v0.pyNegMC(), v0.pzPosMC() + v0.pzNegMC()}, o2::constants::physics::MassPhoton);
    if (pdgCode == 3122 || pdgCode == -3122)
      rapidity = RecoDecay::y(std::array{v0.pxPosMC() + v0.pxNegMC(), v0.pyPosMC() + v0.pyNegMC(), v0.pzPosMC() + v0.pzNegMC()}, o2::constants::physics::MassLambda0);

    if (std::abs(rapidity) > 0.5f)
      return;

    double ptmc = std::hypot(v0.pxPosMC() + v0.pxNegMC(), v0.pyPosMC() + v0.pyNegMC(), v0.pzPosMC() + v0.pzNegMC());

    // step 1: count number of times this candidate was actually reconstructed
    histos.fill(HIST("hNRecoV0s"), recv0s.size());
    bool hasWrongCollision = false;
    float centrality = 100.5f;
    bool hasBeenAcceptablyTracked = false;
    bool hasBeenFoundAny = false;
    bool hasBeenFound = false;
    bool hasBeenAcceptablyTracked_ITSOnly = false;
    bool hasBeenAcceptablyTracked_TPCOnly = false;
    bool hasBeenAcceptablyTracked_ITSTPC = false;
    bool hasBeenAcceptablyTracked_ITSTPCOuter = false;
    bool hasBeenFound_ITSOnly = false;
    bool hasBeenFound_TPCOnly = false;
    bool hasBeenFound_ITSTPC = false;
    bool hasBeenFound_ITSTPCOuter = false;
    bool hasBeenAcceptablyTracked_TPCOthers = false;
    bool hasTPCOthersFound = false;
    bool hasoneTPCOnly = false;
    bool hasoneTPCOnlyFound = false;
    int nCandidatesWithTPC = 0;
    float DCADau_Acctrk_ITSonly =-100;
    float DCADau_Acctrk_TPConly =-100;
    float DCADau_Acctrk_ITSTPC = -100;
    float DCADau_Acctrk_ITSTPCOuter = -100;
    float DCADau_Found_ITSonly =-100;
    float DCADau_Found_TPConly =-100;
    float DCADau_Found_ITSTPC = -100;
    float DCADau_Found_ITSTPCOuter = -100;
    float RecoRadius_Acctrk_ITSonly =-100;
    float RecoRadius_Acctrk_TPConly =-100;
    float RecoRadius_Acctrk_ITSTPC = -100;
    float RecoRadius_Acctrk_ITSTPCOuter = -100;
    float RecoRadius_Found_ITSonly =-100;
    float RecoRadius_Found_TPConly =-100;
    float RecoRadius_Found_ITSTPC = -100;
    float RecoRadius_Found_ITSTPCOuter = -100;

    for (auto& recv0 : recv0s) {

      histos.fill(HIST("hv0Type"), recv0.v0Type());

      if (recv0.v0Type() == 0) // skip v0 for cascades analysis
        continue; 

      if (selStandardV0 && recv0.v0Type() != 1)
        continue; 

      // de-reference daughter track extras
      auto pTrack = recv0.posTrackExtra_as<dauTracks>();
      auto nTrack = recv0.negTrackExtra_as<dauTracks>();

      // skip ITS-only for simplicity
      if (skipITSonly) {
        if (!pTrack.hasTPC() || !nTrack.hasTPC())
          continue;
      }

      // define properties for this V0
      bool pTrackOK = false, nTrackOK = false; // tracks are acceptably tracked

      // tracks detailed info
      bool pTrack_isITSonly = false, nTrack_isITSonly = false; 
      bool pTrack_isTPConly = false, nTrack_isTPConly = false;
      bool pTrack_isITSTPC = false, nTrack_isITSTPC = false;
      bool pTrack_isITSTPCOuter = false, nTrack_isITSTPCOuter = false;
      
      // Detailed analysis level
      bool topoV0RadiusOK = false, topoV0RadiusMaxOK = false, topoV0CosPAOK = false, topoDcaPosToPVOK = false, topoDcaNegToPVOK = false, topoDcaV0DauOK = false;

      if (recv0.has_straCollision()) {
        auto coll = recv0.straCollision_as<recoStraCollisions>();
        int mcCollID_fromCollision = coll.straMCCollisionId();
        int mcCollID_fromV0 = recv0.straMCCollisionId();
        if (mcCollID_fromCollision != mcCollID_fromV0) {
          hasWrongCollision = true;
        } else {
          // if this is a correctly collision-associated V0, take centrality from here
          // N.B.: this could still be an issue if collision <-> mc collision is imperfect
          centrality = coll.centFT0C();
        }

        if (recv0.isFound()) {
          hasBeenFoundAny = true; // includes also ITS-only, checked before skipITSonly check
        }

        if (
          (pTrack.hasTPC() && pTrack.hasITS()) ||     // full global track
          (pTrack.hasTPC() && pTrack.hasTOF()) ||     // TPC + TOF is accepted
          (pTrack.hasTPC() && pTrack.hasTRD()) ||     // TPC + TRD is accepted
          (!pTrack.hasTPC() && pTrack.itsNCls() >= 6) // long ITS-only
        ) {
          pTrackOK = true; // for this V0 only
        }
        if (
          (nTrack.hasTPC() && nTrack.hasITS()) ||
          (nTrack.hasTPC() && nTrack.hasTOF()) || // TPC + TOF is accepted
          (nTrack.hasTPC() && nTrack.hasTRD()) || // TPC + TRD is accepted
          (!nTrack.hasTPC() && nTrack.itsNCls() >= 6)) {
          nTrackOK = true; // for this V0 only
        }
        if (pTrackOK && nTrackOK)
          hasBeenAcceptablyTracked = true;

        // Getting tracking properties
        pTrack_isITSonly = pTrack.hasITS() && !pTrack.hasTPC() && !pTrack.hasTOF() && !pTrack.hasTRD();
        nTrack_isITSonly = nTrack.hasITS() && !nTrack.hasTPC() && !nTrack.hasTOF() && !nTrack.hasTRD();
        pTrack_isTPConly = !pTrack.hasITS() && pTrack.hasTPC() && !pTrack.hasTOF() && !pTrack.hasTRD();
        nTrack_isTPConly = !nTrack.hasITS() && nTrack.hasTPC() && !nTrack.hasTOF() && !nTrack.hasTRD();
        pTrack_isITSTPC = pTrack.hasITS() && pTrack.hasTPC() && !pTrack.hasTOF() && !pTrack.hasTRD();      
        nTrack_isITSTPC = nTrack.hasITS() && nTrack.hasTPC() && !nTrack.hasTOF() && !nTrack.hasTRD();
        pTrack_isITSTPCOuter = pTrack.hasITS() && pTrack.hasTPC() && (pTrack.hasTRD() || pTrack.hasTOF());      
        nTrack_isITSTPCOuter = nTrack.hasITS() && nTrack.hasTPC() && (nTrack.hasTRD() || nTrack.hasTOF());

        if (pTrack_isITSonly && nTrack_isITSonly){
          hasBeenAcceptablyTracked_ITSOnly = true;
          DCADau_Acctrk_ITSonly = recv0.dcaV0daughters();
          RecoRadius_Acctrk_ITSonly = recv0.v0radius();
        
          if (recv0.isFound()){
            hasBeenFound_ITSOnly = true;
            RecoRadius_Found_ITSonly = recv0.v0radius();
            DCADau_Found_ITSonly = recv0.dcaV0daughters();
          }
        }
        if (pTrack_isTPConly && nTrack_isTPConly){ // TPConly
          hasBeenAcceptablyTracked_TPCOnly = true;
          DCADau_Acctrk_TPConly = recv0.dcaV0daughters();
          RecoRadius_Acctrk_TPConly = recv0.v0radius();
          if (recv0.isFound()){
            hasBeenFound_TPCOnly = true;
            RecoRadius_Found_TPConly = recv0.v0radius();
            DCADau_Found_TPConly = recv0.dcaV0daughters();
          }
        }
        if (pTrack_isITSTPC && nTrack_isITSTPC){
          hasBeenAcceptablyTracked_ITSTPC = true;
          DCADau_Acctrk_ITSTPC = recv0.dcaV0daughters();
          RecoRadius_Acctrk_ITSTPC = recv0.v0radius();
          if (recv0.isFound()){
            hasBeenFound_ITSTPC = true;
            RecoRadius_Found_ITSTPC = recv0.v0radius();
            DCADau_Found_ITSTPC = recv0.dcaV0daughters();
          }
        }
        if (pTrack_isITSTPCOuter && nTrack_isITSTPCOuter){
          hasBeenAcceptablyTracked_ITSTPCOuter = true;
          DCADau_Acctrk_ITSTPCOuter = recv0.dcaV0daughters();
          RecoRadius_Acctrk_ITSTPCOuter = recv0.v0radius();
          if (recv0.isFound()){
            hasBeenFound_ITSTPCOuter = true;
            RecoRadius_Found_ITSTPCOuter = recv0.v0radius();
            DCADau_Found_ITSTPCOuter = recv0.dcaV0daughters();
          }
        }

        // TPC specific:
        if ((pTrack_isITSTPCOuter || pTrack_isITSTPC) && (nTrack_isITSTPCOuter || nTrack_isITSTPC)){ // TPC+something
          hasBeenAcceptablyTracked_TPCOthers = true;
          if (recv0.isFound()){
            hasTPCOthersFound = true;
          }
        }
        if ((pTrack_isTPConly && !nTrack_isTPConly) || (!pTrack_isTPConly && nTrack_isTPConly)){ // just one TPConly track
          hasoneTPCOnly = true;
          if (recv0.isFound()){
            hasoneTPCOnlyFound = true;
            }
        }
        
        // cross-check correctness of new getter
        if (pTrack.hasITSTracker() && (pTrack.hasITS() && pTrack.itsChi2PerNcl() < -1e-3)) {
          LOGF(fatal, "Positive track: inconsistent outcome of ITS tracker getter and explicit check!");
        }
        if (nTrack.hasITSTracker() && (pTrack.hasITS() && nTrack.itsChi2PerNcl() < -1e-3)) {
          LOGF(fatal, "Negative track: inconsistent outcome of ITS tracker getter and explicit check!");
        }
        if (pTrack.hasITSAfterburner() && (pTrack.hasITS() && pTrack.itsChi2PerNcl() > -1e-3)) {
          LOGF(fatal, "Positive track: inconsistent outcome of ITS tracker getter and explicit check!");
        }
        if (nTrack.hasITSAfterburner() && (pTrack.hasITS() && nTrack.itsChi2PerNcl() > -1e-3)) {
          LOGF(fatal, "Negative track: inconsistent outcome of ITS tracker getter and explicit check!");
        }

        // Cross-checking consistency: found should be a subset of nTrack, pTrack OK
        histos.fill(HIST("hFoundVsTracksOK"), nTrackOK && pTrackOK, recv0.isFound());

        // encode conditions of tracks
        uint8_t positiveTrackCode = ((uint8_t(pTrack.hasTPC()) << hasTPC) |
                                     (uint8_t(pTrack.hasITSTracker()) << hasITSTracker) |
                                     (uint8_t(pTrack.hasITSAfterburner()) << hasITSAfterburner) |
                                     (uint8_t(pTrack.hasTRD()) << hasTRD) |
                                     (uint8_t(pTrack.hasTOF()) << hasTOF));

        uint8_t negativeTrackCode = ((uint8_t(nTrack.hasTPC()) << hasTPC) |
                                     (uint8_t(nTrack.hasITSTracker()) << hasITSTracker) |
                                     (uint8_t(nTrack.hasITSAfterburner()) << hasITSAfterburner) |
                                     (uint8_t(nTrack.hasTRD()) << hasTRD) |
                                     (uint8_t(nTrack.hasTOF()) << hasTOF));

        if (pTrackOK && nTrackOK && ptmc > 1.0 && ptmc < 1.1) {
          // this particular V0 reco entry has been acceptably tracked. Do bookkeeping
          histos.fill(HIST("h2dTrackPropAcceptablyTracked"), positiveTrackCode, negativeTrackCode, centrality);
        }

        // determine if this V0 would go to analysis or not
        if (recv0.isFound() && pTrackOK && nTrackOK) { // hack to avoid type check; only interested in found type 1
          // at this stage, this should be REALLY mostly unique (unless you switch skipITSonly to false or so)
          // ... but we will cross-check this assumption (hNRecoV0sWithTPC, h2dPtVsCentrality_FoundInLoop)
          if (pTrack.hasTPC() && !pTrack.hasITS() && !pTrack.hasTRD() && !pTrack.hasTOF()) {
            LOGF(info, "Positive track is TPC only and this is a found V0. Puzzling!");
          }
          if (nTrack.hasTPC() && !nTrack.hasITS() && !nTrack.hasTRD() && !nTrack.hasTOF()) {
            LOGF(info, "Negative track is TPC only and this is a found V0. Puzzling!");
          }

          nCandidatesWithTPC++;
          hasBeenFound = true;
          histos.fill(HIST("h2dPtVsCentrality_FoundInLoop"), centrality, ptmc);
          if (ptmc > 1.0 && ptmc < 1.1) {
            histos.fill(HIST("h2dTrackPropFound"), positiveTrackCode, negativeTrackCode, centrality);
          }

          uint64_t selMap = v0data::computeReconstructionBitmap(recv0, pTrack, nTrack, coll, recv0.yLambda(), recv0.yK0Short(), v0Selections);

          // Consider in all cases
          selMap = selMap | (uint64_t(1) << v0data::selConsiderK0Short) | (uint64_t(1) << v0data::selConsiderLambda) | (uint64_t(1) << v0data::selConsiderAntiLambda);

          // selection checker: ensure this works on subset of actual svertexer-findable
          bool validTrackProperties = v0Selections->verifyMask(selMap, maskTrackProperties) && pTrackOK && nTrackOK;
          bool validTopology = v0Selections->verifyMask(selMap, maskTopological);

          uint64_t thisSpeciesMask = maskK0ShortSpecific;
          if (pdgCode == 3122)
            thisSpeciesMask = maskLambdaSpecific;
          if (pdgCode == -3122)
            thisSpeciesMask = maskAntiLambdaSpecific;
          // add other species masks as necessary

          bool validThisSpecies = v0Selections->verifyMask(selMap, thisSpeciesMask);

          // specific selection (not cumulative)
          topoV0RadiusOK = v0Selections->verifyMask(selMap, uint64_t(1) << v0data::selRadius);
          topoV0RadiusMaxOK = v0Selections->verifyMask(selMap, uint64_t(1) << v0data::selRadiusMax);
          topoV0CosPAOK = v0Selections->verifyMask(selMap, uint64_t(1) << v0data::selCosPA);
          topoDcaPosToPVOK = v0Selections->verifyMask(selMap, uint64_t(1) << v0data::selDCAPosToPV);
          topoDcaNegToPVOK = v0Selections->verifyMask(selMap, uint64_t(1) << v0data::selDCANegToPV);
          topoDcaV0DauOK = v0Selections->verifyMask(selMap, uint64_t(1) << v0data::selDCAV0Dau);
          // trackTPCRowsOK = v0Selections->verifyMask(selMap, (uint64_t(1) << v0data::selPosGoodTPCTrack) | (uint64_t(1) << v0data::selNegGoodTPCTrack) );

          // uint64_t tpcPidMask = (uint64_t(1) << v0data::selTPCPIDPositivePion) | (uint64_t(1) << v0data::selTPCPIDNegativePion);
          // if(pdgCode==3122)
          //   tpcPidMask = (uint64_t(1) << v0data::selTPCPIDPositiveProton) | (uint64_t(1) << v0data::selTPCPIDNegativePion);
          // if(pdgCode==-3122)
          //   tpcPidMask = (uint64_t(1) << v0data::selTPCPIDPositivePion) | (uint64_t(1) << v0data::selTPCPIDNegativeProton);

          // trackTPCPIDOK = v0Selections->verifyMask(selMap, tpcPidMask);

          // Broad level
          if (validTrackProperties) {
            histos.fill(HIST("h2dPtVsCentrality_PassesTrackQuality"), centrality, ptmc);
            if (ptmc > 1.0 && ptmc < 1.1) {
              histos.fill(HIST("h2dTrackPropAnalysisTracks"), positiveTrackCode, negativeTrackCode, centrality);
            }
          }
          if (validTrackProperties && validTopology) {
            histos.fill(HIST("h2dPtVsCentrality_PassesTopological"), centrality, ptmc);
            if (ptmc > 1.0 && ptmc < 1.1) {
              histos.fill(HIST("h2dTrackPropAnalysisTopo"), positiveTrackCode, negativeTrackCode, centrality);
            }
          }
          if (validTrackProperties && validTopology && validThisSpecies) {
            histos.fill(HIST("h2dPtVsCentrality_PassesThisSpecies"), centrality, ptmc);
            if (ptmc > 1.0 && ptmc < 1.1) {
              histos.fill(HIST("h2dTrackPropAnalysisSpecies"), positiveTrackCode, negativeTrackCode, centrality);
            }
          }

          // topological
          if (validTrackProperties && validThisSpecies && topoV0RadiusOK)
            histos.fill(HIST("h2dPtVsCentrality_V0Radius"), centrality, ptmc);
          if (validTrackProperties && validThisSpecies && topoV0RadiusMaxOK)
            histos.fill(HIST("h2dPtVsCentrality_V0RadiusMax"), centrality, ptmc);
          if (validTrackProperties && validThisSpecies && topoV0CosPAOK)
            histos.fill(HIST("h2dPtVsCentrality_V0CosPA"), centrality, ptmc);
          if (validTrackProperties && validThisSpecies && topoDcaPosToPVOK)
            histos.fill(HIST("h2dPtVsCentrality_DcaPosToPV"), centrality, ptmc);
          if (validTrackProperties && validThisSpecies && topoDcaNegToPVOK)
            histos.fill(HIST("h2dPtVsCentrality_DcaNegToPV"), centrality, ptmc);
          if (validTrackProperties && validThisSpecies && topoDcaV0DauOK)
            histos.fill(HIST("h2dPtVsCentrality_DcaV0Dau"), centrality, ptmc);
        }
      } else {
        continue;
      }
    }
    // Calculate decay radius
    float V0MCRadius = RecoDecay::sqrtSumOfSquares(v0.xMC(), v0.yMC());

    // Filling histograms
    histos.fill(HIST("hNRecoV0sWithTPC"), nCandidatesWithTPC);

    // Major check 1: Findable versus found in some capacity
    histos.fill(HIST("h2dPtVsCentrality_Findable"), centrality, ptmc);

    // N. of reconstructed 
    histos.fill(HIST("h3dPtVsCentralityNRecoV0s_Findable"), centrality, ptmc, recv0s.size());

    // Findable radius
    histos.fill(HIST("h2dPtVsCentrality_Findable_Radius"), centrality, ptmc, V0MCRadius);

    if (hasBeenAcceptablyTracked) {
      histos.fill(HIST("h2dPtVsCentrality_AcceptablyTracked"), centrality, ptmc);
    }
    if (hasBeenFoundAny) {
      histos.fill(HIST("h2dPtVsCentrality_FoundAny"), centrality, ptmc);
    }
    if (hasBeenFound) {
      histos.fill(HIST("h2dPtVsCentrality_Found"), centrality, ptmc);
    }
    if (hasWrongCollision) {
      histos.fill(HIST("hNRecoV0sWrongColl"), recv0s.size());
    }
    // Acceptably tracked specific
    if (hasBeenAcceptablyTracked_ITSOnly){
      float RadiusReso_ITSOnly = TMath::Abs(V0MCRadius-RecoRadius_Acctrk_ITSonly);

      histos.fill(HIST("h2dPtVsCentrality_AcceptTracked_ITSOnly"), centrality, ptmc);
      histos.fill(HIST("h3dPtVsCentralityRadius_AcceptTracked_ITSOnly"), centrality, ptmc, V0MCRadius); 
      histos.fill(HIST("h3dPtVsCentralityDCADau_AcceptTracked_ITSOnly"), centrality, ptmc, DCADau_Acctrk_ITSonly); 
      histos.fill(HIST("hRadiusResolution_AcceptTracked_ITSOnly"), RadiusReso_ITSOnly); 
    }
    if (hasBeenAcceptablyTracked_TPCOnly){
      float RadiusReso_TPCOnly = TMath::Abs(V0MCRadius-RecoRadius_Acctrk_TPConly);

      histos.fill(HIST("h2dPtVsCentrality_AcceptTracked_TPCOnly"), centrality, ptmc);
      histos.fill(HIST("h3dPtVsCentralityRadius_AcceptTracked_TPCOnly"), centrality, ptmc, V0MCRadius);
      histos.fill(HIST("h3dPtVsCentralityDCADau_AcceptTracked_TPCOnly"), centrality, ptmc, DCADau_Acctrk_TPConly);
      histos.fill(HIST("hRadiusResolution_AcceptTracked_TPCOnly"), RadiusReso_TPCOnly);  
      histos.fill(HIST("h3dPtVsCentralityNRecoV0s_AcceptablyTracked_TPCOnly"), centrality, ptmc, recv0s.size());

      if (hasWrongCollision) {
        histos.fill(HIST("h3dPtVsCentralityNRecoV0sWrongColl_AcceptablyTracked_TPCOnly"), centrality, ptmc, recv0s.size());
      }
      
    }
    if (hasBeenAcceptablyTracked_TPCOthers){
      histos.fill(HIST("h3dPtVsCentralityNRecoV0s_AcceptablyTracked_TPCOther"), centrality, ptmc, recv0s.size());
      if (hasWrongCollision) {
        histos.fill(HIST("h3dPtVsCentralityNRecoV0sWrongColl_AcceptablyTracked_TPCOther"), centrality, ptmc, recv0s.size());
      }
    }
    if (hasoneTPCOnly){
      histos.fill(HIST("h3dPtVsCentralityNRecoV0s_AcceptablyTracked_anyTPCOnly"), centrality, ptmc, recv0s.size());
      if (hasWrongCollision) {
        histos.fill(HIST("h3dPtVsCentralityNRecoV0sWrongColl_AcceptablyTracked_anyTPCOnly"), centrality, ptmc, recv0s.size());
      }
    }
    if (hasBeenAcceptablyTracked_ITSTPC){
      float RadiusReso_ITSTPC = TMath::Abs(V0MCRadius-RecoRadius_Acctrk_ITSTPC);

      histos.fill(HIST("h2dPtVsCentrality_AcceptTracked_ITSTPC"), centrality, ptmc);
      histos.fill(HIST("h3dPtVsCentralityRadius_AcceptTracked_ITSTPC"), centrality, ptmc, V0MCRadius);
      histos.fill(HIST("h3dPtVsCentralityDCADau_AcceptTracked_ITSTPC"), centrality, ptmc, DCADau_Acctrk_ITSTPC); 
      histos.fill(HIST("hRadiusResolution_AcceptTracked_ITSTPC"), RadiusReso_ITSTPC);
    }
    if (hasBeenAcceptablyTracked_ITSTPCOuter){
      float RadiusReso_ITSTPCOuter = TMath::Abs(V0MCRadius-RecoRadius_Acctrk_ITSTPCOuter);

      histos.fill(HIST("h2dPtVsCentrality_AcceptTracked_ITSTPCOuter"), centrality, ptmc);
      histos.fill(HIST("h3dPtVsCentralityRadius_AcceptTracked_ITSTPCOuter"), centrality, ptmc, V0MCRadius);
      histos.fill(HIST("h3dPtVsCentralityDCADau_AcceptTracked_ITSTPCOuter"), centrality, ptmc, DCADau_Acctrk_ITSTPCOuter); 
      histos.fill(HIST("hRadiusResolution_AcceptTracked_ITSTPCOuter"), RadiusReso_ITSTPCOuter);
    }
    
    // Founds specific
    if (hasBeenFound_ITSOnly){
      float RadiusReso_ITSOnly = TMath::Abs(V0MCRadius-RecoRadius_Found_ITSonly);

      histos.fill(HIST("h2dPtVsCentrality_Found_ITSOnly"), centrality, ptmc);
      histos.fill(HIST("hRadiusResolution_Found_ITSOnly"), RadiusReso_ITSOnly);
      histos.fill(HIST("h3dPtVsCentralityRadius_Found_ITSOnly"), centrality, ptmc, V0MCRadius); 
      histos.fill(HIST("h3dPtVsCentralityDCADau_Found_ITSOnly"), centrality, ptmc, DCADau_Found_ITSonly); 
    }
    if (hasBeenFound_TPCOnly){
      float RadiusReso_TPCOnly = TMath::Abs(V0MCRadius-RecoRadius_Found_TPConly);

      histos.fill(HIST("h2dPtVsCentrality_Found_TPCOnly"), centrality, ptmc);
      histos.fill(HIST("hRadiusResolution_Found_TPCOnly"), RadiusReso_TPCOnly);
      histos.fill(HIST("h3dPtVsCentralityRadius_Found_TPCOnly"), centrality, ptmc, V0MCRadius); 
      histos.fill(HIST("h3dPtVsCentralityDCADau_Found_TPCOnly"), centrality, ptmc, DCADau_Found_TPConly); 
      histos.fill(HIST("h3dPtVsCentralityNRecoV0s_Found_TPCOnly"), centrality, ptmc, recv0s.size());

      if (hasWrongCollision) {
        histos.fill(HIST("h3dPtVsCentralityNRecoV0sWrongColl_Found_TPCOnly"), centrality, ptmc, recv0s.size());
      }
    }
    if (hasTPCOthersFound){
      histos.fill(HIST("h3dPtVsCentralityNRecoV0s_Found_TPCOther"), centrality, ptmc, recv0s.size());
      if (hasWrongCollision) {
        histos.fill(HIST("h3dPtVsCentralityNRecoV0sWrongColl_Found_TPCOther"), centrality, ptmc, recv0s.size());
      }
    }
    if (hasoneTPCOnlyFound){
      histos.fill(HIST("h3dPtVsCentralityNRecoV0s_Found_anyTPCOnly"), centrality, ptmc, recv0s.size());
      if (hasWrongCollision) {
        histos.fill(HIST("h3dPtVsCentralityNRecoV0sWrongColl_Found_anyTPCOnly"), centrality, ptmc, recv0s.size());
      }
    }
    if (hasBeenFound_ITSTPC){
      float RadiusReso_ITSTPC = TMath::Abs(V0MCRadius-RecoRadius_Found_ITSTPC);

      histos.fill(HIST("h2dPtVsCentrality_Found_ITSTPC"), centrality, ptmc);
      histos.fill(HIST("hRadiusResolution_Found_ITSTPC"), RadiusReso_ITSTPC);
      histos.fill(HIST("h3dPtVsCentralityRadius_Found_ITSTPC"), centrality, ptmc, V0MCRadius); 
      histos.fill(HIST("h3dPtVsCentralityDCADau_Found_ITSTPC"), centrality, ptmc, DCADau_Found_ITSTPC); 
    }
    if (hasBeenFound_ITSTPCOuter){
      float RadiusReso_ITSTPCOuter = TMath::Abs(V0MCRadius-RecoRadius_Found_ITSTPCOuter);

      histos.fill(HIST("h2dPtVsCentrality_Found_ITSTPCOuter"), centrality, ptmc);
      histos.fill(HIST("hRadiusResolution_Found_ITSTPCOuter"), RadiusReso_ITSTPCOuter);
      histos.fill(HIST("h3dPtVsCentralityRadius_Found_ITSTPCOuter"), centrality, ptmc, V0MCRadius); 
      histos.fill(HIST("h3dPtVsCentralityDCADau_Found_ITSTPCOuter"), centrality, ptmc, DCADau_Found_ITSTPCOuter); 
    }
  }

  PROCESS_SWITCH(findableStudy, processEvents, "process collision counters", true);
  PROCESS_SWITCH(findableStudy, processDebugCrossCheck, "process debug cross-check of V0 with TPC-only", true);
  PROCESS_SWITCH(findableStudy, processDebugCrossCheckMC, "process debug cross-check of V0 with TPC-only, MC version", false);
  PROCESS_SWITCH(findableStudy, processV0s, "process V0s", true);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<findableStudy>(cfgc)};
}
