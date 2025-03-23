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

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;

// For MC association in pre-selection
using LabeledTracksExtra = soa::Join<aod::TracksExtra, aod::McTrackLabels>;

struct v0assoqa {

  Preslice<aod::V0s> perCollision = o2::aod::v0::collisionId;

  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};
  
  Configurable<int> PDGCodePosDau{"PDGCodePosDau", -11, "select PDG code of positive daughter track"};
  Configurable<int> PDGCodeNegDau{"PDGCodeNegDau", 11, "select PDG code of negative daughter track"};
  Configurable<int> PDGCodeMother{"PDGCodeMother", 22, "select PDG code of mother particle"};

  void init(InitContext const&)
  {
    histos.add("h2dPtVsCollAssoc", "h2dPtVsCollAssoc", kTH2D, {{100, 0.0f, 10.0f}, {2, -0.5f, 1.5f}});
    histos.get<TH2>(HIST("h2dPtVsCollAssoc"))->GetYaxis()->SetBinLabel(1, "Wrong collision");
    histos.get<TH2>(HIST("h2dPtVsCollAssoc"))->GetYaxis()->SetBinLabel(2, "Correct collision");

    histos.add("hNRecoV0s", "hNRecoV0s", kTH1D, {{50, -0.5, 49.5f}});

    // Add new histogram for mcpt of particleMC
    histos.add("hMcPtParticleMC", "hMcPtParticleMC", kTH1D, {{100, 0.0f, 10.0f}});
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

  void processBuildMCAssociated(soa::Join<aod::Collisions, aod::McCollisionLabels> const& collisions, aod::V0s const& v0table, LabeledTracksExtra const&, aod::McParticles const& particlesMC)
  {
    std::map<int, int> mcV0Counts;
    std::set<int> filledMcPt; // Set to keep track of filled particleMC ids
    for (auto& collision : collisions) {
      auto V0s = v0table.sliceBy(perCollision, collision.globalIndex());
      
      for (auto const& v0 : V0s) {

        auto lNegTrack = v0.template negTrack_as<LabeledTracksExtra>();
        auto lPosTrack = v0.template posTrack_as<LabeledTracksExtra>();
        float mcpt = -1.0;

        if (lNegTrack.has_mcParticle() && lPosTrack.has_mcParticle()) {
          auto lMCNegTrack = lNegTrack.template mcParticle_as<aod::McParticles>();
          auto lMCPosTrack = lPosTrack.template mcParticle_as<aod::McParticles>();  
               
          int v0id = FindCommonMotherFrom2Prongs(lMCPosTrack, lMCNegTrack, PDGCodePosDau, PDGCodeNegDau, PDGCodeMother, particlesMC);

          if (v0id > 0) {
            auto mcv0 = particlesMC.iteratorAt(v0id);
            mcV0Counts[v0id]++;
            
            int correctMcCollisionIndex = mcv0.mcCollisionId();
            mcpt = mcv0.pt();

            // Fill only if this particleMC has not been filled before
            if (filledMcPt.find(v0id) == filledMcPt.end()) {
              histos.fill(HIST("hMcPtParticleMC"), mcpt);
              filledMcPt.insert(v0id);
            }

            bool collisionAssociationOK = false;
            if (correctMcCollisionIndex > -1 && correctMcCollisionIndex == collision.mcCollisionId()) {
              collisionAssociationOK = true;
            }
            histos.fill(HIST("h2dPtVsCollAssoc"), mcpt, collisionAssociationOK);
          }
        }
      }

      for (const auto& [v0id, count] : mcV0Counts) {
        histos.fill(HIST("hNRecoV0s"), count);
      }
    }
  }

  PROCESS_SWITCH(v0assoqa, processBuildMCAssociated, "Check wrong v0-to-collision association", true);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{adaptAnalysisTask<v0assoqa>(cfgc)};
}